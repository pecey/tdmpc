# Experiment 1: Without TEAs.

project = "tdmpc-hopper-hop"
groups = {"PV": "hopper-hop-state-using-ac-repeats-with-policy-and-value-1768876764",
          "P": "hopper-hop-state-using-ac-repeats-with-policy-and-without-value-1768876764",
          "V": "hopper-hop-state-using-ac-repeats-without-policy-and-with-value-1768876765",
          "None": "hopper-hop-state-using-ac-repeats-without-policy-and-value-1768876764"
          }
results_dir = "hopper-hop-using-ac-repeats-1768876764"
title="Hopper Hop"


project = "tdmpc-quadruped-walk"
groups = {"PV": "quadruped-walk-state-using-ac-repeats-with-policy-and-value-1768876764",
          "P": "quadruped-walk-state-using-ac-repeats-with-policy-and-without-value-1768876764",
          "V": "quadruped-walk-state-using-ac-repeats-without-policy-and-with-value-1768876765",
          "None": "quadruped-walk-state-using-ac-repeats-without-policy-and-value-1768876764"
          }
results_dir = "quadruped-walk-using-ac-repeats-1768876764"
title="Quadruped Walk"


project = "tdmpc-cheetah-run"
groups = {"PV": "cheetah-run-state-using-ac-repeats-with-policy-and-value-1768876764",
          "P": "cheetah-run-state-using-ac-repeats-with-policy-and-without-value-1768876764",
          "V": "cheetah-run-state-using-ac-repeats-without-policy-and-with-value-1768876765",
          "None": "cheetah-run-state-using-ac-repeats-without-policy-and-value-1768876764"
          }
results_dir = "cheetah-run-using-ac-repeats-1768876764"
title="Cheetah Run"


# Experiment 2: With TEAs

project = "tdmpc-hopper-hop"
groups = {"AR": "hopper-hop-state-using-ac-repeats-with-policy-and-value-1768876764",
          "TEA": "hopper-hop-state-using-tea-with-policy-and-value-5x-1769012240",
          }
results_dir = "hopper-hop-using-teas-1768876764"
title="Hopper Hop"

project = "tdmpc-quadruped-walk"
groups = {"AR": "quadruped-walk-state-using-ac-repeats-with-policy-and-value-1768876764",
          "TEA": "quadruped-walk-state-using-tea-with-policy-and-value-5x-1769012241"
          }
results_dir = "quadruped-walk-using-teas-1768876764"
title="Quadruped Walk"

project = "tdmpc-cheetah-run"
groups = {"AR": "cheetah-run-state-using-ac-repeats-with-policy-and-value-1768876764",
          "TEA": "cheetah-run-state-using-tea-with-policy-and-value-5x-1769016325"
          }
results_dir = "cheetah-run-using-teas-1768876764"
title="Cheetah Run"

# Experiment 3: With TEAs but without policy and value function

project = "tdmpc-hopper-hop"
groups = {"AR": "hopper-hop-state-using-ac-repeats-without-policy-and-value-1768876764",
          "TEA": "hopper-hop-state-using-tea-without-policy-and-value-1768940730",
          "TEA (d=10)" : "hopper-hop-state-using-tea-without-policy-and-value-d-10-1769024551",
          }
results_dir = "hopper-hop-using-teas-wo-policy-and-value-function-1768876764"
title="Hopper Hop"

project = "tdmpc-quadruped-walk"
groups = {"AR": "quadruped-walk-state-using-ac-repeats-without-policy-and-value-1768876764",
          "TEA": "quadruped-walk-state-using-tea-without-policy-and-value-1768940730",
          "TEA (d=10)" : "quadruped-walk-state-using-tea-without-policy-and-value-d-10-1769024551"
          }
results_dir = "quadruped-walk-using-teas-wo-policy-and-value-function-1768876764"
title="Quadruped Walk"

project = "tdmpc-cheetah-run"
groups = {"AR": "cheetah-run-state-using-ac-repeats-without-policy-and-value-1768876764",
          "TEA": "cheetah-run-state-using-tea-without-policy-and-value-1768940730",
          "TEA (d=10)": "cheetah-run-state-using-tea-without-policy-and-value-d-10-1769024551"
          }
results_dir = "cheetah-run-using-teas-wo-policy-and-value-function-1768876764"
title="Cheetah Run"
