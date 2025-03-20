import pygame
import sys

# 定义常用颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 定义棋子类，保存棋子的名称、颜色以及在棋盘上的位置（以格子坐标表示）
class Piece:
    def __init__(self, name, color, pos):
        self.name = name      # 棋子的汉字，例如 "车", "马"
        self.color = color    # "red" 或 "black"
        self.pos = pos        # 格子坐标 (列, 行)，例如 (0, 0)
        self.is_selected = False  # 标记棋子是否被选中

# 棋盘窗口类，用于绘制棋盘和初始化棋子
class ChessBoardWindow:
    def __init__(self, headless=False, agent_model=None, mcts_simulations=10):
        # 必须初始化的核心逻辑
        self.headless = headless
        self.agent_model = agent_model
        self.mcts_simulations = mcts_simulations

        # 初始化棋子数据
        self.cell_size = 60  # 定义棋盘每个格子的大小
        self.pieces = []  # 存储所有棋子
        self.init_pieces()
        self.current_turn = "red"  # 'red' 为人类，'black' 为 AI
        self.selected_piece = None  # 记录当前选中的棋子
        
        # 如果 headless=True，则直接返回，不创建 GUI 相关资源
        if self.headless:
            return
        
        # GUI 相关初始化
        self.screen = pygame.display.get_surface()
        self.width, self.height = self.screen.get_size()
        
        # 计算棋盘位置（使其居中）
        board_width = self.cell_size * 8
        board_height = self.cell_size * 9
        self.board_origin = ((self.width - board_width) // 2, (self.height - board_height) // 2)
        
        # 初始化字体
        self.font = pygame.font.SysFont("WenQuanYi Zen Hei", 24)
    def init_pieces(self):
        """
        初始化棋子到标准布局位置，预留后续接口用于棋子状态更新和交互逻辑的扩展
        """
        # 黑方（上方）棋子布局
        self.pieces.append(Piece("车", "black", (0, 0)))
        self.pieces.append(Piece("马", "black", (1, 0)))
        self.pieces.append(Piece("相", "black", (2, 0)))
        self.pieces.append(Piece("仕", "black", (3, 0)))
        self.pieces.append(Piece("将", "black", (4, 0)))
        self.pieces.append(Piece("仕", "black", (5, 0)))
        self.pieces.append(Piece("相", "black", (6, 0)))
        self.pieces.append(Piece("马", "black", (7, 0)))
        self.pieces.append(Piece("车", "black", (8, 0)))
        self.pieces.append(Piece("炮", "black", (1, 2)))
        self.pieces.append(Piece("炮", "black", (7, 2)))
        self.pieces.append(Piece("卒", "black", (0, 3)))
        self.pieces.append(Piece("卒", "black", (2, 3)))
        self.pieces.append(Piece("卒", "black", (4, 3)))
        self.pieces.append(Piece("卒", "black", (6, 3)))
        self.pieces.append(Piece("卒", "black", (8, 3)))
        
        # 红方（下方）棋子布局
        self.pieces.append(Piece("车", "red", (0, 9)))
        self.pieces.append(Piece("马", "red", (1, 9)))
        self.pieces.append(Piece("象", "red", (2, 9)))
        self.pieces.append(Piece("士", "red", (3, 9)))
        self.pieces.append(Piece("帅", "red", (4, 9)))
        self.pieces.append(Piece("士", "red", (5, 9)))
        self.pieces.append(Piece("象", "red", (6, 9)))
        self.pieces.append(Piece("马", "red", (7, 9)))
        self.pieces.append(Piece("车", "red", (8, 9)))
        self.pieces.append(Piece("炮", "red", (1, 7)))
        self.pieces.append(Piece("炮", "red", (7, 7)))
        self.pieces.append(Piece("兵", "red", (0, 6)))
        self.pieces.append(Piece("兵", "red", (2, 6)))
        self.pieces.append(Piece("兵", "red", (4, 6)))
        self.pieces.append(Piece("兵", "red", (6, 6)))
        self.pieces.append(Piece("兵", "red", (8, 6)))
    
    def draw_board(self, surface):
        """
        绘制棋盘：包括网格线、宫廷对角线以及“楚河汉界”文字。
        在 headless 模式下，该方法可选择不调用（或仅用于调试）。
        """
        ox, oy = self.board_origin
        
        for col in range(9):
            x = ox + col * self.cell_size
            if col == 0 or col == 8:
                pygame.draw.line(surface, BLACK, (x, oy), (x, oy + self.cell_size * 9), 2)
            else:
                pygame.draw.line(surface, BLACK, (x, oy), (x, oy + self.cell_size * 4), 2)
                pygame.draw.line(surface, BLACK, (x, oy + self.cell_size * 5), (x, oy + self.cell_size * 9), 2)
        
        for row in range(10):
            y = oy + row * self.cell_size
            pygame.draw.line(surface, BLACK, (ox, y), (ox + self.cell_size * 8, y), 2)
        
        pygame.draw.line(surface, BLACK, (ox + 3 * self.cell_size, oy), 
                         (ox + 5 * self.cell_size, oy + 2 * self.cell_size), 2)
        pygame.draw.line(surface, BLACK, (ox + 5 * self.cell_size, oy), 
                         (ox + 3 * self.cell_size, oy + 2 * self.cell_size), 2)
        pygame.draw.line(surface, BLACK, (ox + 3 * self.cell_size, oy + 7 * self.cell_size), 
                         (ox + 5 * self.cell_size, oy + 9 * self.cell_size), 2)
        pygame.draw.line(surface, BLACK, (ox + 5 * self.cell_size, oy + 7 * self.cell_size), 
                         (ox + 3 * self.cell_size, oy + 9 * self.cell_size), 2)
        
        river_font = pygame.font.SysFont("WenQuanYi Zen Hei", 36)
        text_chu = river_font.render("楚河", True, BLACK)
        text_han = river_font.render("汉界", True, BLACK)
        text_chu_rect = text_chu.get_rect(center=(ox + self.cell_size * 2, oy + self.cell_size * 4.5))
        text_han_rect = text_han.get_rect(center=(ox + self.cell_size * 6, oy + self.cell_size * 4.5))
        surface.blit(text_chu, text_chu_rect)
        surface.blit(text_han, text_han_rect)
    
    def draw_pieces(self, surface):
        """遍历所有棋子，在棋盘上绘制（以圆形和文字表示）"""
        ox, oy = self.board_origin
        for piece in self.pieces:
            col, row = piece.pos
            center_x = ox + col * self.cell_size
            center_y = oy + row * self.cell_size
            radius = self.cell_size // 2 - 5
            pygame.draw.circle(surface, WHITE, (center_x, center_y), radius)
            piece_color = (255, 0, 0) if piece.color == "red" else BLACK
            pygame.draw.circle(surface, piece_color, (center_x, center_y), radius, 2)
            text_surf = self.font.render(piece.name, True, piece_color)
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            surface.blit(text_surf, text_rect)
    
    def get_piece_at(self, pos):
        """返回给定棋盘格坐标 pos 上的棋子，没有则返回 None"""
        for piece in self.pieces:
            if piece.pos == pos:
                return piece
        return None

    def is_valid_move(self, piece, new_pos):
        """
        判断从 piece.pos 移动到 new_pos 是否符合中国象棋的规则
        """
        start_col, start_row = piece.pos
        new_col, new_row = new_pos
        dx = new_col - start_col
        dy = new_row - start_row

        if not (0 <= new_col <= 8 and 0 <= new_row <= 9):
            return False

        if piece.name in ["帅", "将"]:
            if piece.color == "red":
                if not (3 <= new_col <= 5 and 7 <= new_row <= 9):
                    return False
            else:
                if not (3 <= new_col <= 5 and 0 <= new_row <= 2):
                    return False
            if abs(dx) + abs(dy) != 1:
                return False
            for p in self.pieces:
                if p != piece and p.name in ["帅", "将"]:
                    if new_col == p.pos[0]:
                        step = 1 if new_row < p.pos[1] else -1
                        for r in range(new_row + step, p.pos[1], step):
                            if self.get_piece_at((new_col, r)) is not None:
                                break
                        else:
                            return False
            return True

        if piece.name in ["士", "仕"]:
            if abs(dx) != 1 or abs(dy) != 1:
                return False
            if piece.color == "red":
                if not (3 <= new_col <= 5 and 7 <= new_row <= 9):
                    return False
            else:
                if not (3 <= new_col <= 5 and 0 <= new_row <= 2):
                    return False
            return True

        if piece.name in ["象", "相"]:
            if abs(dx) != 2 or abs(dy) != 2:
                return False
            eye = ((start_col + new_col) // 2, (start_row + new_row) // 2)
            if self.get_piece_at(eye) is not None:
                return False
            if piece.color == "red":
                if new_row < 5:
                    return False
            else:
                if new_row > 4:
                    return False
            return True

        if piece.name == "马":
            if (abs(dx), abs(dy)) not in [(2, 1), (1, 2)]:
                return False
            if abs(dx) == 2:
                leg = (start_col + dx // 2, start_row)
            else:
                leg = (start_col, start_row + dy // 2)
            if self.get_piece_at(leg) is not None:
                return False
            return True

        if piece.name == "车":
            if dx != 0 and dy != 0:
                return False
            if dx == 0:
                step = 1 if dy > 0 else -1
                for r in range(start_row + step, new_row, step):
                    if self.get_piece_at((start_col, r)) is not None:
                        return False
            else:
                step = 1 if dx > 0 else -1
                for c in range(start_col + step, new_col, step):
                    if self.get_piece_at((c, start_row)) is not None:
                        return False
            return True

        if piece.name == "炮":
            if dx != 0 and dy != 0:
                return False
            target_piece = self.get_piece_at(new_pos)
            if target_piece is None:
                if dx == 0:
                    step = 1 if dy > 0 else -1
                    for r in range(start_row + step, new_row, step):
                        if self.get_piece_at((start_col, r)) is not None:
                            return False
                else:
                    step = 1 if dx > 0 else -1
                    for c in range(start_col + step, new_col, step):
                        if self.get_piece_at((c, start_row)) is not None:
                            return False
                return True
            else:
                count = 0
                if dx == 0:
                    step = 1 if dy > 0 else -1
                    for r in range(start_row + step, new_row, step):
                        if self.get_piece_at((start_col, r)) is not None:
                            count += 1
                else:
                    step = 1 if dx > 0 else -1
                    for c in range(start_col + step, new_col, step):
                        if self.get_piece_at((c, start_row)) is not None:
                            count += 1
                return count == 1

        if piece.name in ["兵", "卒"]:
            if piece.color == "red":
                if start_row > 4:
                    if dx != 0 or dy != -1:
                        return False
                else:
                    if (dy == -1 and dx == 0) or (dy == 0 and abs(dx) == 1):
                        return True
                    else:
                        return False
                return True
            else:
                if start_row < 5:
                    if dx != 0 or dy != 1:
                        return False
                else:
                    if (dy == 1 and dx == 0) or (dy == 0 and abs(dx) == 1):
                        return True
                    else:
                        return False
                return True

        return False

    def get_possible_moves(self, piece):
        moves = []
        col, row = piece.pos
        if piece.name in ["帅", "将"]:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in directions:
                moves.append((col + dx, row + dy))
        elif piece.name in ["士", "仕"]:
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in directions:
                moves.append((col + dx, row + dy))
        elif piece.name in ["象", "相"]:
            directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
            for dx, dy in directions:
                moves.append((col + dx, row + dy))
        elif piece.name == "马":
            directions = [(2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2)]
            for dx, dy in directions:
                moves.append((col + dx, row + dy))
        elif piece.name in ["车", "炮"]:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in directions:
                new_col = col + dx
                new_row = row + dy
                while 0 <= new_col <= 8 and 0 <= new_row <= 9:
                    moves.append((new_col, new_row))
                    new_col += dx
                    new_row += dy
        elif piece.name in ["兵", "卒"]:
            if piece.color == "red":
                moves.append((col, row - 1))
                if row <= 4:
                    moves.append((col - 1, row))
                    moves.append((col + 1, row))
            else:
                moves.append((col, row + 1))
                if row >= 5:
                    moves.append((col - 1, row))
                    moves.append((col + 1, row))
        return moves

    def handle_red_move(self, event):
        mx, my = event.pos
        ox, oy = self.board_origin
        col = (mx - ox + self.cell_size // 2) // self.cell_size
        row = (my - oy + self.cell_size // 2) // self.cell_size

        if col < 0 or col > 8 or row < 0 or row > 9:
            return

        target_piece = self.get_piece_at((col, row))
        if self.selected_piece is None:
            if target_piece and target_piece.color == "red":
                self.selected_piece = target_piece
                target_piece.is_selected = True
        else:
            if target_piece and target_piece.color == "red":
                self.selected_piece.is_selected = False
                self.selected_piece = target_piece
                target_piece.is_selected = True
                return
            if self.is_valid_move(self.selected_piece, (col, row)):
                if target_piece and target_piece.color != "red":
                    self.pieces.remove(target_piece)
                self.selected_piece.pos = (col, row)
                self.current_turn = "black"
            else:
                print("非法走法，请重新选择！")
            self.selected_piece.is_selected = False
            self.selected_piece = None

    def check_game_over(self):
        red_exists = any(piece for piece in self.pieces if piece.name == "帅")
        black_exists = any(piece for piece in self.pieces if piece.name == "将")
        if not red_exists:
            return True, "black"
        if not black_exists:
            return True, "red"
        return False, None

    def show_game_over(self, winner):
        font = pygame.font.SysFont("WenQuanYi Zen Hei", 48)
        msg = "红方胜！" if winner == "red" else "黑方胜！"
        text_surf = font.render("游戏结束: " + msg, True, (255, 0, 0))
        text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_surf, text_rect)
        pygame.display.flip()
        pygame.time.wait(3000)

    def handle_agent_turn(self):
        board_state = self.get_board_state()
        agent_move = self.get_agent_move(board_state)
        if agent_move:
            piece, new_pos = agent_move
            target_piece = self.get_piece_at(new_pos)
            if target_piece and target_piece.color != piece.color:
                self.pieces.remove(target_piece)
            piece.pos = new_pos
        self.current_turn = "red"

    def get_board_state(self):
        return self.pieces

    def encode_board_state(self):
        board = [[0 for _ in range(9)] for _ in range(10)]
        piece_mapping = {
            "车": 1,
            "马": 2,
            "象": 3,
            "相": 3,
            "士": 4,
            "仕": 4,
            "帅": 5,
            "炮": 6,
            "兵": 7,
            "卒": 7,
            "将": 5
        }
        for piece in self.pieces:
            col, row = piece.pos
            value = piece_mapping.get(piece.name, 0)
            if piece.color == "black":
                value = -value
            board[row][col] = value
        return board

    def reset(self):
        self.pieces = []
        self.init_pieces()
        self.current_turn = "red"
        self.selected_piece = None
        return self.encode_board_state()

    def step(self, action):
        start_pos, new_pos = action
        #获取棋子
        piece = self.get_piece_at(start_pos)
        #棋子存在而且是合法的移动
        if piece and self.is_valid_move(piece, new_pos):
            target_piece = self.get_piece_at(new_pos)
            if target_piece and target_piece.color != piece.color:
                self.pieces.remove(target_piece)
            piece.pos = new_pos

            # 切换当前走棋方
            if self.current_turn == "red":
                self.current_turn = "black"
            else:
                self.current_turn = "red"

            game_over, winner = self.check_game_over()
            done = game_over
            reward = 0
            if done:
                reward = 1 if winner == piece.color else -1
            return self.encode_board_state(), reward, done, {"winner": winner}
        else:
            return self.encode_board_state(), -1, False, {"error": "illegal move"}
        

    def apply_move(self, piece, new_pos):
        if self.is_valid_move(piece, new_pos):
            target_piece = self.get_piece_at(new_pos)
            if target_piece and target_piece.color != piece.color:
                self.pieces.remove(target_piece)
            piece.pos = new_pos
            return True
        return False

    def get_agent_move(self, board_state):
        """
        使用传入的 agent_model 结合 MCTS 选择最优动作。
        这里先用 encode_board_state() 获取当前棋盘编码，
        然后构造 policy_value_fn 调用 agent_model 得到先验概率和局面价值，
        最后用 MCTS 选出动作，并转换为 (piece, new_pos) 形式返回。
        """
        # 采用编码状态作为 MCTS 搜索输入
        state = self.encode_board_state()
        
        def pv_fn(s):
            import torch
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                log_probs, value = self.agent_model(s_tensor)
            probs = torch.exp(log_probs).numpy().flatten()
            return {i: probs[i] for i in range(len(probs))}, value.item()
        
        from mcts import MCTS, MCTSNode
        root = MCTSNode(state)
        mcts = MCTS(pv_fn, c_puct=1.0, n_simulations=self.mcts_simulations)
        mcts.run(root)
        action_index, _ = mcts.get_action(root, temperature=1e-3)
        
        # 利用 index_to_move 将动作索引转换为移动 ((start_col, start_row), (end_col, end_row))
        from self_play import index_to_move
        move = index_to_move(action_index)
        start_pos, new_pos = move
        piece = self.get_piece_at(start_pos)
        return (piece, new_pos)

    def run(self):
        if self.headless:
            # 如果处于 headless 模式，则不进入事件循环
            return
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                if self.current_turn == "red" and event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_red_move(event)
            
            if self.current_turn == "black":
                self.handle_agent_turn()
            
            self.screen.fill((210, 180, 140))
            self.draw_board(self.screen)
            self.draw_pieces(self.screen)
            pygame.display.flip()
            clock.tick(60)
            game_over, winner = self.check_game_over()
            if game_over:
                self.show_game_over(winner)
                running = False
