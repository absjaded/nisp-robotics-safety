import mujoco
import numpy as np
from src.nisp.core.contracts import TelemetryState

class L0_Ingestion:
    """
    Layer 0: MuJoCo Adapter
    Maps MuJoCo simulation data to the NISP TelemetryState contract.
    """
    def __init__(self, model: mujoco.MjModel, ee_link_name: str = "ee_link"):
        self.model = model
        self.ee_link_name = ee_link_name
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_link_name)
        if self.ee_id == -1:
            raise ValueError(
                f"L0_Ingestion: body '{ee_link_name}' not found in MuJoCo model. "
                f"Pass ee_link_name= to the constructor."
            )
            
    def ingest(self, data: mujoco.MjData, min_dist: float = 10.0, 
               zone_status: int = 0, closing_velocity: float = 0.0,
               human_aabb_center: np.ndarray = None,
               tcp_velocity: float = None) -> TelemetryState:
        """
        Transforms mjData into a TelemetryState object.
        """
        ee_pos = data.xpos[self.ee_id].copy()

        if human_aabb_center is None:
            human_aabb_center = np.zeros(3)

        if tcp_velocity is None:
            tcp_vel_vec = np.zeros(6)  # [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.ee_id, tcp_vel_vec, 0
            )
            tcp_velocity = np.linalg.norm(tcp_vel_vec[3:])

        return TelemetryState(
            q=data.qpos.copy(),
            q_dot=data.qvel.copy(),
            tau=data.qfrc_actuator.copy(),
            ee_pos=ee_pos,
            min_dist=min_dist,
            zone_status=zone_status,
            sim_time=data.time,
            closing_velocity=closing_velocity,
            human_aabb_center=human_aabb_center,
            tcp_velocity=tcp_velocity
        )

    def process(self, data: mujoco.MjData, **kwargs) -> TelemetryState:
        """
        Standardized interface for the NISP pipeline.
        """
        return self.ingest(data, **kwargs)
