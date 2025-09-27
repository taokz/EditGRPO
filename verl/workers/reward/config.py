# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward config
"""

from dataclasses import dataclass
from typing import Optional, List
from ...utils.reward_score.rad_scores import VALID_REWARD_COMPONENTS  # Import the valid components
from ...utils.reward_score.green_scores import VALID_GREEN_REWARD_COMPONENTS  # Import GREEN components


@dataclass
class RewardConfig:
    reward_type: str = "function"
    compute_score: str = "math"
    reward_server: str = "radgraph"
    type: str = "cot"
    reward_components: Optional[List[str]] = None
    
    def __post_init__(self):
        # Validate type
        if self.type not in {'short', 'cot', 'free_form'}:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of: 'short', 'cot', 'free_form'")
            
        # Validate reward components if specified
        if self.reward_components is not None:
            if not isinstance(self.reward_components, list):
                raise ValueError(f"reward_components must be a list, got {type(self.reward_components)}")
            
            if self.reward_components == ["all"]:
                return
            
            # Choose validation based on compute_score type
            if self.compute_score == "green_server":
                valid_components = VALID_GREEN_REWARD_COMPONENTS
                component_type = "GREEN"
            elif self.compute_score == "rad_server":
                valid_components = VALID_REWARD_COMPONENTS
                component_type = "RAD"
            else:
                # For other compute_score types, skip component validation
                return
                
            invalid_components = [c for c in self.reward_components if c not in valid_components]
            if invalid_components:
                raise ValueError(
                    f"Invalid {component_type} reward components: {invalid_components}. "
                    f"Valid options are: {sorted(valid_components)}"
                )
