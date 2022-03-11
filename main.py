import os

from trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer('./config.yml')
    trainer.train()

    trainer.load_best_model(os.path.join(trainer.log_root, 'best_model.pt'))
    valid_acc = trainer.evaluate_acc()
    print(f'valid accuracy: {valid_acc}')
