import inspect
from pathlib import Path

import CybORG

def get_scenario(name="Scenario2", from_cyborg=True):

    if from_cyborg:
        # scenario_path = inspect.getfile(CybORG)[:-10] + f"/Shared/Scenarios/{self.scenario}.yaml"
        cyborg_path = Path(inspect.getfile(CybORG)).resolve()
        scenario_dir = cyborg_path.parent / "Shared" / "Scenarios"
    else:
        scenario_dir = Path(__file__).resolve().parent.parent / "scenarios"


    scenario_path =  scenario_dir / Path(name).with_suffix(".yaml")
    
    assert scenario_path.exists()
    print(f"Loaded scenario file from {scenario_path}")
    
    return scenario_path
