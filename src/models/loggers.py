from typing import Optional

from pytorch_lightning.loggers import WandbLogger


class CustomWandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        entity: Optional[str] = None,
        tags: Optional[list] = None,
    ):
        super().__init__(
            project=project, save_dir=save_dir, entity=entity, tags=tags, log_model=log_model
        )
