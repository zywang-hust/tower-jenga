"""Task instructions: the competition's wire text and the training data's per-variant text."""

# Competition instructions per task family (https://challenge.tower-benchmark.com/docs/task-prompts/). They differ
# from every training instruction, and the X-2 and X-3 variants share one sentence.
OFFICIAL_INSTRUCTIONS = {
    "task1": "Use both arms cooperatively to transfer the Jenga tower to the designated target position.",
    "task2": "Use both arms cooperatively to transfer the cross-shaped Jenga tower to the designated target position.",
    "task3": "Use both arms cooperatively to stack the Jenga blocks vertically at the designated target position.",
    "task4": "Push the designated block from the full-layer Jenga tower while preserving the required tower structure.",
    "task5": "Use both arms to insert a block into the designated empty slot of the two-block Jenga tower.",
}
# The instruction attribute of every training episode, per variant (runs/audit/r1_summary.json).
TRAIN_INSTRUCTIONS = {
    "task1-2": "Transfer 4 upper side blocks from the source tower onto the destination foundation.",
    "task1-3": "Transfer 6 upper side blocks from the source tower onto the destination foundation.",
    "task2-2": "Transfer 2 centered blocks top-down and reconstruct the cross tower on the destination foundation.",
    "task2-3": "Transfer 3 centered blocks top-down and reconstruct the cross tower on the destination foundation.",
    "task3-2": "Transfer 2 aligned single blocks onto the destination foundation, keeping their centers vertically aligned.",
    "task3-3": "Transfer 3 aligned single blocks onto the destination foundation, keeping their centers vertically aligned.",
    "task4-2": "Push 6 upper blocks off the full tower, one at a time from top to bottom and left to right in each "
    "alternating row.",
    "task4-3": "Push 9 upper blocks off the full tower, one at a time from top to bottom and left to right in each "
    "alternating row.",
    "task5-2": "Insert 1 blocks into the tower center gaps, highest gap first, without destabilizing the tower.",
    "task5-3": "Insert 2 blocks into the tower center gaps, highest gap first, without destabilizing the tower.",
}
