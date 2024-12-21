from train_chat.models.chat_model import ChatModel
from train_chat.utils.data_loader import load_data
from train_chat.utils.helper import set_seed
import yaml

def main(config_path):
    # 設定ファイルを読み込む
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # ランダムシードの設定
    set_seed(config['seed'])

    # データの読み込み
    train_data, val_data = load_data(config['data_path'])

    # モデルの初期化
    model = ChatModel(config['model'])

    # モデルの学習
    model.train(train_data, val_data, config['training'])

    # 学習結果の保存
    model.save(config['output_path'])

if __name__ == "__main__":
    main("config/default_config.yaml")
