"""RecoverVLA interactive runner.

Algorithm:
Target Memory + Belief Search + Candidate Verification + remote VLA decision
+ local safety controller for language-guided dynamic target re-acquisition.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

from .belief_map import BeliefMap, predict_target_state
from .config import CONFIG, RecoverVLAConfig
from .controller import SafetyController
from .decision import RecoverVLADecisionEngine, build_vla_prompt
from .environment_context import build_environment_context
from .logger import RunLogger, annotate_frame
from .models import DecisionPlan, TargetMemory, TargetState
from .observation import observe_vehicle_candidates, update_target_memory, visible_target_observation
from .server_client import RemoteVLAClient
from .simulator import (
    connect_airsim,
    connect_carla,
    get_collision_info,
    get_drone_image,
    get_drone_state,
    set_camera_view,
    spawn_background_traffic,
    spawn_target_vehicle,
)
from .utils import parse_instruction


def _serialize_candidates(candidates):
    serializable = []
    for candidate in candidates or []:
        item = dict(candidate)
        state = item.get("state")
        if hasattr(state, "to_dict"):
            item["state"] = state.to_dict()
        serializable.append(item)
    return serializable


class RecoverVLARunner:
    def __init__(self, config: RecoverVLAConfig = CONFIG):
        self.config = config
        self.remote = RemoteVLAClient(config)
        self.decision_engine = RecoverVLADecisionEngine(config)
        self.belief = BeliefMap(config)
        self.controller = SafetyController(config)
        self.last_latency_ms: Optional[float] = None

    def run_episode(
        self,
        airsim_client,
        airsim_module,
        world,
        carla_module,
        instruction: str,
        target_actor=None,
        target_query: Optional[Dict[str, Any]] = None,
    ) -> None:
        memory = TargetMemory(
            raw_instruction=instruction,
            color=(target_query or {}).get("color"),
            vehicle_type=(target_query or {}).get("vehicle_type"),
            appearance=f"{(target_query or {}).get('color') or ''} {(target_query or {}).get('vehicle_type') or ''}".strip(),
        )
        logger = RunLogger(self.config.log_dir)
        executor = ThreadPoolExecutor(max_workers=1)
        pending: Optional[Future] = None
        pending_meta: Dict[str, Any] = {}
        plan: Optional[DecisionPlan] = None
        next_decision_t = 0.0
        decision_idx = 0
        frame_idx = 0
        previous_state: Optional[TargetState] = None
        last_state: Optional[TargetState] = None

        policy_name = "remote Qwen VLA" if self.config.use_remote_vla else "local rule fallback (no VLA)"
        print(f"  RecoverVLA 启动: Target Memory + Belief Search + Candidate Verification + {policy_name}")
        print(f"  指令: {instruction}")

        try:
            for loop_idx in range(self.config.max_steps):
                now = time.time()
                drone = get_drone_state(airsim_client)
                collision = get_collision_info(airsim_client)
                if collision is not None and getattr(collision, "has_collided", False):
                    print("  检测到碰撞，进入 return_home 兜底")
                    next_decision_t = now

                horizon = self.belief.horizon(self.last_latency_ms)
                predicted = predict_target_state(memory.last_seen, memory.previous_seen, horizon)
                belief_seed = predicted or memory.last_seen
                belief_center = belief_seed.position if belief_seed else drone.position
                belief_heading = belief_seed.heading if belief_seed else drone.heading
                env_context = build_environment_context(
                    world,
                    carla_module,
                    drone.position,
                    belief_center,
                    belief_heading,
                    plan.waypoint if plan else None,
                    self.config,
                )
                belief_region = self.belief.build(memory, predicted, now, env_context)

                bbox_meta, visible_candidates = observe_vehicle_candidates(world, airsim_client, target_actor, self.config)
                observed = visible_target_observation(target_actor, bbox_meta)
                local_candidate = self.decision_engine.verifier.score(bbox_meta, observed, memory, belief_region, visible_candidates)
                memory_updated = update_target_memory(
                    memory,
                    observed,
                    local_candidate,
                    now,
                    update_threshold=self.config.memory_update_threshold,
                )
                if memory_updated:
                    previous_state = last_state
                    last_state = observed
                    predicted = predict_target_state(memory.last_seen, memory.previous_seen, horizon)
                    belief_seed = predicted or memory.last_seen
                    env_context = build_environment_context(
                        world,
                        carla_module,
                        drone.position,
                        belief_seed.position if belief_seed else drone.position,
                        belief_seed.heading if belief_seed else drone.heading,
                        plan.waypoint if plan else None,
                        self.config,
                    )
                    belief_region = self.belief.build(memory, predicted, now, env_context)
                    local_candidate = self.decision_engine.verifier.score(bbox_meta, observed, memory, belief_region, visible_candidates)

                image = get_drone_image(airsim_client, airsim_module)
                if image is None:
                    print("  无法获取无人机图像，等待下一帧")
                    time.sleep(self.config.control_dt)
                    continue

                if pending is not None and pending.done():
                    result = pending.result()
                    self.last_latency_ms = float(result.get("latency_ms") or 0.0)
                    meta = pending_meta
                    pending = None
                    pending_meta = {}
                    plan = self.decision_engine.make_plan(
                        meta["drone"],
                        memory,
                        meta["observed"],
                        meta["predicted"],
                        meta["bbox_meta"],
                        result,
                        plan,
                        world,
                        carla_module,
                        candidate_list=meta.get("visible_candidates"),
                        environment=meta.get("environment"),
                    )
                    set_camera_view(airsim_client, airsim_module, plan.view, self.config)
                    annotated = annotate_frame(meta["image"], plan, meta["bbox_meta"], f"decision {decision_idx:04d}")
                    frame_name = f"decision_{decision_idx:04d}.png"
                    path = logger.save_frame(
                        annotated,
                        frame_name,
                        {
                            "type": "decision",
                            "loop_idx": loop_idx,
                            "decision_idx": decision_idx,
                            "plan": plan.to_dict(),
                            "target_memory": memory.to_prompt_dict(now),
                            "observed": meta["observed"].to_dict() if meta["observed"] else None,
                            "predicted": meta["predicted"].to_dict() if meta["predicted"] else None,
                            "bbox_meta": meta["bbox_meta"],
                            "visible_candidates": _serialize_candidates(meta.get("visible_candidates", [])),
                            "environment": meta.get("environment"),
                            "remote_result": result,
                        },
                        bbox=meta["bbox_meta"].get("bbox"),
                        image_size=meta["bbox_meta"].get("image_size"),
                    )
                    logger.write(
                        {
                            "type": "decision",
                            "loop_idx": loop_idx,
                            "decision_idx": decision_idx,
                            "plan": plan.to_dict(),
                            "frame": frame_name,
                            "latency_ms": self.last_latency_ms,
                            "image_path": path,
                        }
                    )
                    print(
                        f"  [decision {decision_idx:03d}] phase={plan.phase} view={plan.view} "
                        f"found={plan.target_found} conf={plan.confidence:.2f} latency={self.last_latency_ms:.0f}ms"
                    )
                    decision_idx += 1
                    next_decision_t = now + self.config.decision_interval

                if self.config.use_remote_vla and pending is None and now >= next_decision_t:
                    prompt = build_vla_prompt(instruction, drone, memory, belief_region, local_candidate, plan)
                    pending_meta = {
                        "image": image,
                        "drone": drone,
                        "observed": observed,
                        "predicted": predicted,
                        "bbox_meta": bbox_meta,
                        "visible_candidates": visible_candidates,
                        "environment": env_context,
                    }
                    pending = executor.submit(self.remote.predict, image, prompt)
                    next_decision_t = now + self.config.decision_interval
                    print(f"  [request] phase_hint={plan.phase if plan else 'bootstrap'} belief={belief_region.reason}")

                if plan is None:
                    plan = self.decision_engine.make_plan(
                        drone,
                        memory,
                        observed,
                        predicted,
                        bbox_meta,
                        None,
                        None,
                        world,
                        carla_module,
                        candidate_list=visible_candidates,
                        environment=env_context,
                    )
                    set_camera_view(airsim_client, airsim_module, plan.view, self.config)

                # Keep waypoint fresh while waiting for slow remote inference.
                plan = self.decision_engine.make_plan(
                    drone,
                    memory,
                    observed,
                    predicted,
                    bbox_meta,
                    {"action": plan.raw_model},
                    plan,
                    world,
                    carla_module,
                    candidate_list=visible_candidates,
                    environment=env_context,
                )
                duration = min(max(self.config.control_dt, plan.hold_seconds), self.config.decision_interval)
                action = self.controller.action(drone, plan, duration)
                logger.write(
                    {
                        "type": "control",
                        "loop_idx": loop_idx,
                        "plan": plan.to_dict(),
                        "drone": drone.to_dict(),
                        "observed": observed.to_dict() if observed else None,
                        "predicted": predicted.to_dict() if predicted else None,
                        "belief": belief_region.to_dict(),
                        "bbox_meta": bbox_meta,
                        "visible_candidates": _serialize_candidates(visible_candidates),
                        "environment": env_context,
                        "memory_updated": memory_updated,
                        "action": action.to_dict(),
                        "pending_remote": pending is not None,
                    }
                )

                visible = bool(bbox_meta.get("visible"))
                stride = self.config.visible_frame_stride if visible else self.config.invisible_frame_stride
                if self.config.save_control_frames and stride > 0 and loop_idx % stride == 0:
                    annotated = annotate_frame(image, plan, bbox_meta, f"control {frame_idx:04d}")
                    logger.save_frame(
                        annotated,
                        f"control_{frame_idx:04d}.png",
                        {"type": "control_frame", "loop_idx": loop_idx, "plan": plan.to_dict(), "bbox_meta": bbox_meta},
                        bbox=bbox_meta.get("bbox"),
                        image_size=bbox_meta.get("image_size"),
                    )
                    frame_idx += 1

                print(
                    f"  [{loop_idx:04d}] phase={plan.phase} view={plan.view} "
                    f"alt={-drone.position[2]:.1f}m goal={action.distance_3d:.1f}m "
                    f"vx={action.vx:+.2f} vy={action.vy:+.2f} vz={action.vz:+.2f} yaw={action.yaw_rate:+.1f}"
                )
                airsim_client.moveByVelocityBodyFrameAsync(
                    action.vx,
                    action.vy,
                    action.vz,
                    action.duration,
                    drivetrain=airsim_module.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim_module.YawMode(True, action.yaw_rate),
                )
                time.sleep(self.config.control_dt)
        except KeyboardInterrupt:
            print("\n  用户中断任务")
        finally:
            try:
                airsim_client.hoverAsync()
            except Exception:
                pass
            executor.shutdown(wait=False, cancel_futures=True)
            logger.close()
            print(f"  Episode 结束，共 {decision_idx} 次远程高层决策")


def main() -> None:
    config = CONFIG
    print()
    print("RecoverVLA - CARLA-Air language-guided dynamic target re-acquisition")
    print()
    runner = RecoverVLARunner(config)
    print("连接推理服务器 ...")
    if config.use_remote_vla and not runner.remote.check() and config.remote_required:
        print("请先启动远程 model_server.py，或设置 RECOVER_REMOTE_REQUIRED=0 使用本地兜底策略。")
        sys.exit(1)

    print("连接 CARLA-Air ...")
    try:
        carla_client, world, carla_module = connect_carla(config)
    except Exception as exc:
        print(f"无法连接 CARLA-Air: {exc}")
        sys.exit(1)

    print("连接 AirSim ...")
    try:
        airsim_client, airsim_module = connect_airsim(config)
    except Exception as exc:
        print(f"无法连接 AirSim: {exc}")
        sys.exit(1)

    target_actor = None
    actors = []
    traffic = []
    print()
    print("输入任务指令，例如: 找回失踪的白色厢式卡车，它最后向东行驶")
    print("输入 spawn 白色 卡车 可手动生成目标；输入 quit 退出。")

    while True:
        try:
            instruction = input("指令> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not instruction:
            continue
        if instruction.lower() == "quit":
            break

        parsed = parse_instruction(instruction)
        if instruction.lower().startswith("spawn"):
            parts = instruction.split()
            if len(parts) > 1:
                parsed = parse_instruction(" ".join(parts[1:]))
            actor = spawn_target_vehicle(world, carla_module, config, parsed.get("color"), parsed.get("vehicle_type") or "sedan")
            if actor:
                actor.set_autopilot(True)
                target_actor = actor
                actors.append(actor)
                if not traffic:
                    traffic = spawn_background_traffic(world, carla_module, config)
            continue

        if target_actor is None:
            print("  未发现目标车辆，按任务描述自动生成一个目标。")
            target_actor = spawn_target_vehicle(world, carla_module, config, parsed.get("color"), parsed.get("vehicle_type") or "sedan")
            if target_actor:
                target_actor.set_autopilot(True)
                actors.append(target_actor)
                if not traffic:
                    traffic = spawn_background_traffic(world, carla_module, config)
                time.sleep(1.5)

        runner.run_episode(airsim_client, airsim_module, world, carla_module, instruction, target_actor, parsed)

    print("清理资源 ...")
    try:
        airsim_client.armDisarm(False)
        airsim_client.enableApiControl(False)
    except Exception:
        pass
    for actor in actors + traffic:
        try:
            actor.destroy()
        except Exception:
            pass
    try:
        carla_client = None
    except Exception:
        pass
    print("完成")
