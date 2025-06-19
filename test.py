import akioi_2048 as ak
import choose_move as cm
import printer


def main(model_dir, deterministic):
    board = ak.init()
    res = 0
    step = 0
    score = 0
    while not res:
        dir, _ = cm.choose_move(
            model_dir=model_dir, board=board, deterministic=deterministic
        )
        new_board, delta_score, res = ak.step(board, dir)
        if new_board != board:
            step += 1
        board = new_board
        score += delta_score
    return (board, res, score, step)


if __name__ == "__main__":
    model_dir = "model"
    deterministic = False
    num = 10
    max_score = 0
    for i in range(num):
        final_board, res, score, step = main(model_dir, deterministic)
        res = ("continue", "win", "lose")[res]
        printer.print_table(final_board)
        print("\n")
        output = [["result", "score", "step"], [res, score, step]]
        printer.print_table(output)
        print("\n\n")

        max_score = max(score, max_score)
    print(max_score)
