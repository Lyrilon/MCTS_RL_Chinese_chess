import argparse
import time
from mcts import Mcts
from PolicyValueNet import PolicyValueNet
from chessboard import Chess_board
from self_play import selfplay  # 假定 selfplay 类已定义

class CChessTrainer:
    def __init__(self, train_epoch, batch_size):
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.data_buffer = []
        self.log_file = open("training_log.txt", "w")
        self.board = Chess_board()
        self.PV_net = PolicyValueNet(num_gpus=1, num_of_res_block=19)
        self.mcts = Mcts("RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr", 
                         self.PV_net, search_threads=16)

    def selfplay(self):
        # 返回 selfplay 数据和每局步数
        sp = selfplay(play_times=1, mcts_search_round=100, temperature=1.0, num_gpu=1)
        return sp.selfplay_n_times()

    def policy_update(self):
        # 占位，采样 self.data_buffer 进行训练更新
        pass

    def run(self):
        batch_iter = 0
        start_time = time.time()
        print("[Train CChess] -> Training Start ({} Epochs)".format(self.train_epoch))
        try:
            total_data_len = 0
            while batch_iter <= self.train_epoch:
                batch_iter += 1
                play_data, episode_len = self.selfplay()
                print("[Train CChess] -> Batch {}/{}; Episode Length: {}; Iteration: {}".format(
                    batch_iter, self.train_epoch, episode_len, batch_iter))
                extend_data = []
                for state, mcts_prob, winner in play_data:
                    states_data = self.mcts.state_to_positions(state)
                    extend_data.append((states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                total_data_len += len(extend_data)
                self.log_file.write("time: {} \t total_data_len: {}\n".format(time.time() - start_time, total_data_len))
                self.log_file.flush()
                print("training data_buffer len:", len(self.data_buffer))
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
            self.log_file.close()
            self.PV_net.save()
            print("[Train CChess] -> Training Finished, Took {} s".format(time.time() - start_time))
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving model...")
            self.log_file.close()
            self.PV_net.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'selfplay'], default='train')
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = CChessTrainer(train_epoch=args.train_epoch, batch_size=args.batch_size)
        trainer.run()
    else:
        sp_instance = selfplay(play_times=10, mcts_search_round=100, temperature=1.0, num_gpu=1)
        play_data, episode_data = sp_instance.selfplay_n_times()
        print("Selfplay finished. Episodes:", episode_data)
