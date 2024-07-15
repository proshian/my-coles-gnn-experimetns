import sys
import logging

from tqdm import tqdm
import hydra
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path=None, config_name=None)
def main(conf: DictConfig):
    dm = hydra.utils.instantiate(conf.data_module)

    train_loader = dm.train_dataloader()

    total_size = 0
    for batch in tqdm(train_loader):
        total_size += sys.getsizeof(batch)
    print(f"{total_size = }")
    # 2296112

 

if __name__ == '__main__':
    main()
