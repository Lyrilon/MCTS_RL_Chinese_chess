from chessboard import Chess_board
from mcts import Mcts
class AiHumanPlay():
    def __init__(self,mcts_round,temperature,PV_net):
        self.playouts = mcts_round
        self.board = Chess_board()
        self.PV_net = PV_net
        self.temperature = temperature
        self.mcts_tree = Mcts("RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr",self.PV_net,search_threads=8)
    def play_begin(self):
        self.board.reset_board()
        self.mcts_tree.reload()
        while not self.board.over:
            round = 0
            if self.board.turn == 'r':
                act = input()
                self.board.move(act)
                round += 1
            else:
                mcts_res = self.mcts_tree.mcts(self.board.turn,self.mcts_search_round,self.temperature,restrict_round=0)
                action_choosen, action_probs ,win_prob = mcts_res
                self.board.move(action_choosen)
                round+=1
        print("winner is {}".format(self.board.winner))

