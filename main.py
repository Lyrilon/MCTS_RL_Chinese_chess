from self_play import selfplay

s = selfplay(play_times=1,mcts_search_round=400,temperature=1,num_gpu=1)

s.selfplay_n_times()


