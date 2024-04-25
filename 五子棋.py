import pygame
import sys

# 棋型的评估分数
shape_score = [(50, (0, 1, 1, 0, 0)),
               (50, (0, 0, 1, 1, 0)),
               (200, (1, 1, 0, 1, 0)),
               (500, (0, 0, 1, 1, 1)),
               (500, (1, 1, 1, 0, 0)),
               (5000, (0, 1, 1, 1, 0)),
               (5000, (0, 1, 0, 1, 1, 0)),
               (5000, (0, 1, 1, 0, 1, 0)),
               (5000, (1, 1, 1, 0, 1)),
               (5000, (1, 1, 0, 1, 1)),
               (5000, (1, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 1, 0)),
               (5000, (0, 1, 1, 1, 1)),
               (500000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]

# 颜色定义
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
GRID_COLOR = (100, 100, 100)
BOARD_COLOR = (249, 214, 91)

# 棋盘参数
GRID_SIZE = 30  # 棋格大小
HEIGHT = 14  # 棋盘高度
WIDTH = 14  # 棋盘宽度
MARGIN = 30  # 边距
CHESS_SIZE = 13  # 棋子大小

# 棋子颜色
BLACK = 1
WHITE = 2

# ai参数：小于1时偏向进攻，大于1时偏向防守
ratio = 0.7
# 最大搜索深度
MAX_DEPTH = 2
next_point = [0, 0]

list1 = []  # ai
list2 = []  # 人
list3 = []  # all

# 窗口参数
WINDOW_WIDTH = WIDTH * GRID_SIZE + 2 * MARGIN
WINDOW_HEIGHT = HEIGHT * GRID_SIZE + 2 * MARGIN

# 初始化Pygame
pygame.init()

# 设置窗口大小
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("五子棋")

# 初始化棋盘状态
board_state = [[0] * (WIDTH + 1) for _ in range(HEIGHT + 1)]
current_player = 1  # 黑棋为1


# 棋盘绘制
def draw_checkerboard(screen):
    # 填充背景色
    screen.fill(BOARD_COLOR)

    # 绘制棋盘网格
    for i in range(WIDTH + 1):
        pygame.draw.line(screen, GRID_COLOR, (MARGIN + i * GRID_SIZE, MARGIN),
                         (MARGIN + i * GRID_SIZE, WINDOW_HEIGHT - MARGIN), 1)

    for i in range(HEIGHT + 1):
        pygame.draw.line(screen, GRID_COLOR, (MARGIN, MARGIN + i * GRID_SIZE),
                         (WINDOW_WIDTH - MARGIN, MARGIN + i * GRID_SIZE), 1)

    # 绘制特殊位置（星位和天元）
    special_position = ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11))
    for pos in special_position:
        pygame.draw.circle(screen, BLACK_COLOR, (MARGIN + pos[1] * GRID_SIZE, MARGIN + pos[0] * GRID_SIZE), 5)

    # 绘制棋子
    for row in range(HEIGHT + 1):
        for col in range(WIDTH + 1):
            if board_state[row][col] == 1:
                pygame.draw.circle(screen, BLACK_COLOR, (MARGIN + col * GRID_SIZE, MARGIN + row * GRID_SIZE), CHESS_SIZE)
            if board_state[row][col] == 2:
                pygame.draw.circle(screen, WHITE_COLOR, (MARGIN + col * GRID_SIZE, MARGIN + row * GRID_SIZE), CHESS_SIZE)

    pygame.display.flip()


# 检查游戏是否结束
def check_game_over(board_state):
    # 检查水平方向是否连成五子
    for row in range(HEIGHT + 1):
        for col in range(WIDTH - 3):
            if board_state[row][col] != 0 and \
                    board_state[row][col] == board_state[row][col + 1] == board_state[row][col + 2] == \
                    board_state[row][col + 3] == board_state[row][col + 4]:
                return board_state[row][col]

    # 检查垂直方向是否连成五子
    for col in range(WIDTH + 1):
        for row in range(HEIGHT - 3):
            if board_state[row][col] != 0 and \
                    board_state[row][col] == board_state[row + 1][col] == board_state[row + 2][col] == \
                    board_state[row + 3][col] == board_state[row + 4][col]:
                return board_state[row][col]

    # 检查主对角线方向是否连成五子
    for row in range(HEIGHT - 3):
        for col in range(WIDTH - 3):
            if board_state[row][col] != 0 and \
                    board_state[row][col] == board_state[row + 1][col + 1] == board_state[row + 2][col + 2] == \
                    board_state[row + 3][col + 3] == board_state[row + 4][col + 4]:
                return board_state[row][col]

    # 检查副对角线方向是否连成五子
    for row in range(HEIGHT - 3):
        for col in range(WIDTH, 3, -1):
            if board_state[row][col] != 0 and \
                    board_state[row][col] == board_state[row + 1][col - 1] == board_state[row + 2][col - 2] == \
                    board_state[row + 3][col - 3] == board_state[row + 4][col - 4]:
                return board_state[row][col]

    # 如果没有任何一方获胜，则返回0表示平局
    return 0


# 获得当前玩家的落子位置
def get_player_move():
    global event
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row = int((pos[1] - MARGIN + 0.5 * GRID_SIZE) // GRID_SIZE)
                col = int((pos[0] - MARGIN + 0.5 * GRID_SIZE) // GRID_SIZE)
                if 0 <= row <= HEIGHT and 0 <= col <= WIDTH:
                    if board_state[row][col] == 0:
                        return row, col


# 计算评估函数值
def evaluation(is_ai, my_list, enemy_list):
    total_score = 0

    if is_ai:
        my_color = BLACK
        enemy_color = WHITE
    else:
        my_color = WHITE
        enemy_color = BLACK

    # 计算当前玩家的得分
    my_score = calculate_score(my_list, enemy_list, my_color)

    # 计算对手的得分
    enemy_score = calculate_score(enemy_list, my_list, enemy_color) * ratio

    # 最终得分为当前玩家得分减去对手得分
    total_score = my_score - enemy_score

    return total_score


# 计算得分
def calculate_score(player_list, enemy_list, player_color):
    total_score = 0
    score_all_arr = []  # 存储已经计算过的得分形状

    # 遍历每个位置
    for pt in player_list:
        m, n = pt

        # 在四个方向上计算得分
        for x_direct, y_direct in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            score = cal_score(m, n, x_direct, y_direct, enemy_list, player_list, score_all_arr)
            total_score += score

    return total_score


# 计算某个方向上的得分
def cal_score(m, n, x_direct, y_direct, enemy_list, my_list, score_all_arr):
    add_score = 0  # 加分项
    max_score_shape = (0, None)  # 最大得分形状及其得分

    # 判断是否已经计算过此方向上的得分形状
    for item in score_all_arr:
        for pt in item[1]:
            if m == pt[0] and n == pt[1] and x_direct == item[2][0] and y_direct == item[2][1]:
                return 0

    # 在当前位置向左右方向上查找得分形状
    for offset in range(-5, 1):
        pos = []
        for i in range(0, 6):
            if (m + (i + offset) * x_direct, n + (i + offset) * y_direct) in enemy_list:
                pos.append(2)
            elif (m + (i + offset) * x_direct, n + (i + offset) * y_direct) in my_list:
                pos.append(1)
            else:
                pos.append(0)
        tmp_shape_5 = tuple(pos[:5])
        tmp_shape_6 = tuple(pos)

        # 遍历所有模型，寻找匹配的形状
        for score, shape in shape_score:
            if tmp_shape_5 == shape or tmp_shape_6 == shape:
                # 更新最大得分形状
                if score > max_score_shape[0]:
                    max_score_shape = (
                        score, [(m + (j + offset) * x_direct, n + (j + offset) * y_direct) for j in range(6)],
                        (x_direct, y_direct))

    # 计算相交的得分
    if max_score_shape[1] is not None:
        for item in score_all_arr:
            for pt1 in item[1]:
                for pt2 in max_score_shape[1]:
                    if pt1 == pt2 and max_score_shape[0] > 10 and item[0] > 10:
                        add_score += item[0] + max_score_shape[0]

        # 将当前得分形状添加到已计算列表中
        score_all_arr.append(max_score_shape)

    return add_score + max_score_shape[0]


def negamax(is_ai, depth, alpha, beta):
    # 游戏是否结束或者是否达到了搜索的最大深度
    if check_game_over(board_state) or depth == 0:
        return evaluation(is_ai, list1 if is_ai else list2, list2 if is_ai else list1)

    # 获取所有可落子的位置
    blank_list = [(i, j) for i in range(HEIGHT + 1) for j in range(WIDTH + 1) if board_state[i][j] == 0]

    # 对可落子的位置进行排序，以提高剪枝效率
    order(blank_list)

    # 初始化最佳值
    best_value = float('-inf')

    # 遍历每一个候选步
    for next_step in blank_list:
        row, col = next_step

        # 如果要评估的位置没有相邻的子，则不去评估，减少计算
        if not has_neighbor(next_step):
            continue

        # 尝试落子
        board_state[row][col] = BLACK if is_ai else WHITE
        if is_ai:
            list1.append((row, col))
        else:
            list2.append((row, col))

        # 递归调用negamax
        value = -negamax(not is_ai, depth - 1, -beta, -alpha)

        # 恢复棋盘状态
        board_state[row][col] = 0
        if is_ai:
            list1.remove((row, col))
        else:
            list2.remove((row, col))

        # 更新最佳值和alpha
        best_value = max(best_value, value)
        alpha = max(alpha, value)

        # alpha-beta剪枝
        if alpha >= beta:
            break

    return best_value


#  离最后落子的邻居位置最有可能是最优点
def order(blank_list):
    last_pt = list3[-1]
    for _ in blank_list:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (last_pt[0] + i, last_pt[1] + j) in blank_list:
                    blank_list.remove((last_pt[0] + i, last_pt[1] + j))
                    blank_list.insert(0, (last_pt[0] + i, last_pt[1] + j))


def has_neighbor(pt):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (pt[0] + i, pt[1] + j) in list3:
                return True
    return False


# 执行搜索
def ai_play():
    # 初始化最佳位置
    best_move = None
    best_value = float('-inf')

    # 获取所有可落子的位置
    blank_list = [(i, j) for i in range(HEIGHT + 1) for j in range(WIDTH + 1) if board_state[i][j] == 0]

    # 对可落子的位置进行排序，以提高剪枝效率
    order(blank_list)

    # 执行搜索
    for next_step in blank_list:
        row, col = next_step

        # 如果要评估的位置没有相邻的子，则不去评估，减少计算
        if not has_neighbor(next_step):
            continue

        # 尝试落子
        board_state[row][col] = BLACK
        list1.append((row, col))

        # 递归调用 negamax
        value = -negamax(False, MAX_DEPTH - 1, float('-inf'), float('inf'))

        # 恢复棋盘状态
        board_state[row][col] = 0
        list1.remove((row, col))

        # 更新最佳位置和最佳值
        if value > best_value:
            best_value = value
            best_move = next_step

    # 返回最佳位置
    return best_move


if __name__ == '__main__':
    # 游戏主循环
    running = True
    while running:
        # 检查事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 绘制棋盘
        draw_checkerboard(screen)

        # 玩家回合
        if current_player == 1:
            row, col = get_player_move()
            if row is not None and col is not None:
                board_state[row][col] = 1
                list2.append((row, col))
                list3.append((row, col))
                current_player = 2  # 切换到白棋
                # 检查游戏是否结束
                if check_game_over(board_state):
                    running = False

        # AI回合
        else:
            row, col = ai_play()
            if row is not None and col is not None:
                board_state[row][col] = 2
                list1.append((row, col))
                list3.append((row, col))
                current_player = 1  # 切换回黑棋
                # 检查游戏是否结束
                if check_game_over(board_state):
                    running = False

        # 更新屏幕显示
        pygame.display.flip()

    # 退出游戏
    pygame.quit()
    sys.exit()
