import mujoco
import numpy as np
import pytest
from src.nisp.layers.l0_ingestion import L0_Ingestion
from src.nisp.core.contracts import TelemetryState

def test_l0_ingestion_basic():
    # Create a minimal MuJoCo model
    xml = """
    <mujoco>
        <worldbody>
            <body name="ee_link" pos="1 1 1">
                <joint type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # Step to initialize
    mujoco.mj_step(model, data)
    
    ingestor = L0_Ingestion(model, ee_link_name="ee_link")
    state = ingestor.ingest(data, min_dist=0.5, zone_status=1)
    
    assert isinstance(state, TelemetryState)
    assert state.sim_time == data.time
    assert state.min_dist == 0.5
    assert state.zone_status == 1
    # Check that ndarrays were converted to tuples (from our contracts.py fix)
    assert isinstance(state.q, tuple)
    assert len(state.ee_pos) == 3
    assert np.allclose(state.ee_pos, data.xpos[ingestor.ee_id])

if __name__ == "__main__":
    pytest.main([__file__])
