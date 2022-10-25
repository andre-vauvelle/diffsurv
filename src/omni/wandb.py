import re

import wandb


def change_name_run():
    api = wandb.Api()
    run_ids = ["2p6j9jlo", "1m1g5beu", "g4uxng4x"]
    name = run_ids[0]

    def change_name(run_id):
        run = api.run(f"qndre/diffsurv/{run_id}")
        names = list(run.summary.keys())
        for n in names:
            if not re.match(r"(^_)", n):
                new_name = n.replace("_", "/")
                run.summary[new_name] = run.summary[n]
                del run.summary[n]
        run.summary.update()

    [change_name(id) for id in run_ids]
