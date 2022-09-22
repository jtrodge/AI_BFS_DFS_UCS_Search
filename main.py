import random
import queue  # used for search algorithm
import itertools  # used for tie between nodes with equal costs, turns it into FIFO for ties
from collections import deque
from copy import deepcopy

# for comparison to goal state
# Can change anytime
goalUCS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
goalBFS = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
goalDFS = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
# A queue to perform the BFS search.
q = deque()
# A set to avoid reaching the previously visited state.
visited = set()
# Using a list in python as a stack to perform the DFS search.
stack = []
# A set to avoid reaching the previously visited state.
visited = set()
# to make priority queue FIFO when there is a tie on priority number
tie = itertools.count()
nodesExpanded = 0
maxNodesInQueue = 0
goalDepth = 0

def main():
    # prompt user for puzzle choice and store decision
    print("Jason's 8-puzzle solver.")
    choice = input('Type "1" to list 10 states out of all possible states:\n'
                   'Type "2" to enter the current state and make an action which returns the resulting state.\n'
                   'Type "3" to enter the initial state of your puzzle.\n'
                   'Type "4" for a random initial state of your puzzle\n'
                   'Choice: ')
    # randomly pick 10 out of all possible states
    if choice == "1":
        allPossibleStates()
    # Prompts user for a current state and action; returns the resulting state
    elif choice == "2":
        puzzle = userGeneratePuzzle()
        action = input('\nAction: ')
        action = int(action)
        print(swap(puzzle, action))
    # if user wants to create puzzle, call function for that
    elif choice == "3":
        puzzle = userGeneratePuzzle()
    elif choice == "4":
        arr = [7, 2, 4, 5, 0, 6, 8, 3, 1]
        new_array = random.sample(arr, len(arr))
        puzzle = new_array
    # prompt for algorithm choice
    print("Now, choose your algorithm to solve")
    print("1. Uniform Cost Search")
    print("2. Breadth First Search")
    print("3. Depth First Search")
    # store algorithm decision
    algChoice = input()
    print("\n")
    print(puzzle)
    if algChoice == "1":
        generalSearch(puzzle, algChoice)
        print("To solve this problem the search algorithm expanded a total of " + str(nodesExpanded) + " nodes")
        print("The maximum number of nodes in the queue at any one time was " + str(maxNodesInQueue))
        print("The depth of the goal was " + str(goalDepth))
        print("The initial state: ", puzzle)
        print("The goal state: ", goalUCS)
    elif algChoice == "2":
        newPuzzle = [[puzzle[0], puzzle[1], puzzle[2]], [puzzle[3],
                     puzzle[4], puzzle[5]], [puzzle[6], puzzle[7], puzzle[8]]]
        bfs(newPuzzle, goalBFS)
        print("The initial state: ", puzzle)
        print("The goal state: ", goalBFS)
    elif algChoice == "3":
        newPuzzle = [[puzzle[0], puzzle[1], puzzle[2]], [puzzle[3], puzzle[4], puzzle[5]],
                     [puzzle[6], puzzle[7], puzzle[8]]]
        dfs(newPuzzle, goalDFS)
        print("The initial state: ", puzzle)
        print("The goal state: ", goalDFS)
    else:
        print("invalid input, uniform cost search inbound\n")
        generalSearch(puzzle, uniformCost, "1")
        print("To solve this problem the search algorithm expanded a total of " + str(nodesExpanded) + " nodes")
        print("The maximum number of nodes in the queue at any one time was " + str(maxNodesInQueue))
        print("The depth of the goal was " + str(goalDepth))
        print("The initial state: ", puzzle)
        print("The goal state: ", goalUCS)

    return

# Prints 10 random states of all possibilities
def allPossibleStates():
    print("10 randomly selected states:")
    arr = [7,2,4,5,0,6,8,3,1]
    for i in range(0,10):
        new_array = random.sample(arr, len(arr))
        print(i + 1,'.) ', new_array)

# swaps the positions in a list
def swap(list, pos):
    list[(pos)], list[((pos+1))] = list[((pos+1))], list[(pos)]
    return list

# BFS
# print BFS
def printBFS(tempA):
    for i in tempA:
        for j in i:
            print(j, end=" ")
        print()
    print()


def bfsLeft(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if j - 1 >= 0:
                    tempB[i][j - 1], tempB[i][j] = tempB[i][j], tempB[i][j - 1]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    q.append(tempB)
                    return

def bfsRight(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if j + 1 < len(tempB):
                    tempB[i][j + 1], tempB[i][j] = tempB[i][j], tempB[i][j + 1]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    q.append(tempB)
                    return

def bfsUp(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if i - 1 >= 0:
                    tempB[i - 1][j], tempB[i][j] = tempB[i][j], tempB[i - 1][j]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    q.append(tempB)
                    return

def bfsDown(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if i + 1 < len(tempB):
                    tempB[i + 1][j], tempB[i][j] = tempB[i][j], tempB[i + 1][j]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    q.append(tempB)
                    return

def bfs(tempA, tempB):
    q.append(tempA)
    count = 0
    while len(q) != 0:
        count += 1;
        now = q.popleft()
        printBFS(now)
        if now == tempB:
            print("Reached destination at iteration ", count)
            return
        for i in range(len(now)):
            for j in range(len(now)):
                if now[i][j] == 0:
                    bfsLeft(now)
                    bfsUp(now)
                    bfsDown(now)
                    bfsRight(now)
        if (len(q) == 0):
            print("Failed -- puzzle unsolvable")

# DFS
# print DFS
def printDFS(tempA):
    for i in tempA:
        for j in i:
            print(j, end=" ")
        print()
    print()

def dfsLeft(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if j - 1 >= 0:
                    tempB[i][j - 1], tempB[i][j] = tempB[i][j], tempB[i][j - 1]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    stack.append(tempB)
                    return

def dfsRight(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if j + 1 < len(tempB):
                    tempB[i][j + 1], tempB[i][j] = tempB[i][j], tempB[i][j + 1]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    stack.append(tempB)
                    return

def dfsUp(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if i - 1 >= 0:
                    tempB[i - 1][j], tempB[i][j] = tempB[i][j], tempB[i - 1][j]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    stack.append(tempB)
                    return

def dfsDown(tempA):
    tempB = deepcopy(tempA);
    for i in range(len(tempB)):
        for j in range(len(tempB)):
            if tempA[i][j] == 0:
                if i + 1 < len(tempB):
                    tempB[i + 1][j], tempB[i][j] = tempB[i][j], tempB[i + 1][j]
                    tempC = tuple(map(tuple, tempB))
                    if tempC in visited:
                        return
                    visited.add(tempC)
                    stack.append(tempB)
                    return

def dfs(tempA, tempB):
    stack.append(tempA)
    count = 0
    while len(stack) != 0:
        count += 1;
        now = stack.pop()
        printDFS(now)
        if now == tempB:
            print("Reached destination at iteration ", count)
            return
        for i in range(len(now)):
            for j in range(len(now)):
                if now[i][j] == 0:
                    dfsRight(now)
                    dfsUp(now)
                    dfsDown(now)
                    dfsLeft(now)
        if not stack:
            print("Failed -- puzzle unsolvable")

#UCS
# finds where blank is in list
def getBlankIndex(p):
    for i in range(len(p)):
        if p[i] == 0:
            return i

# takes a puzzle state (list) as parameter, checks if operation can be performed on that state and returns (state after operation, cost of operator) if so
# returns original state if operation is not possible
def moveUpUCS(p, initCost):
    # set cost
    cost = 1
    blankIndex = getBlankIndex(p)
    # if moving the blank up would put it above the board, return the current puzzle state
    if (blankIndex - 3) < 0:
        return (p, cost)
    # else, return a copy of the board where the blank (0) has been swapped with the number above it
    newBoard = p.copy()
    newBoard[blankIndex] = newBoard[blankIndex - 3]
    newBoard[blankIndex - 3] = 0
    return (newBoard, cost + initCost)


# logic is the same for other moveBlank operators
def moveDownUCS(p, initCost):
    cost = 1
    blankIndex = getBlankIndex(p)
    if (blankIndex + 3) >= len(p):
        return (p, cost)
    blankIndex = getBlankIndex(p)
    newBoard = p.copy()
    newBoard[blankIndex] = newBoard[blankIndex + 3]
    newBoard[blankIndex + 3] = 0
    return (newBoard, cost + initCost)


def moveRightUCS(p, initCost):
    cost = 0.5
    blankIndex = getBlankIndex(p)
    if (blankIndex % 3) == 2:
        return (p, cost)
    blankIndex = getBlankIndex(p)
    newBoard = p.copy()
    newBoard[blankIndex] = newBoard[blankIndex + 1]
    newBoard[blankIndex + 1] = 0
    return (newBoard, cost + initCost)


def moveLeftUCS(p, initCost):
    cost = 2
    blankIndex = getBlankIndex(p)
    if (blankIndex % 3) == 0:
        return (p, cost)
    blankIndex = getBlankIndex(p)
    newBoard = p.copy()
    newBoard[blankIndex] = newBoard[blankIndex - 1]
    newBoard[blankIndex - 1] = 0
    return (newBoard, cost + initCost)


# checks if current puzzle state is the goal state
def isGoalState(p):
    if p == goalUCS:
        return True
    return False


# takes a node of form (cost, tie value, puzzle state) and returns list of tuples of (puzzle states, cost of operator)
def expand(node):
    expansion = [moveUpUCS(node[2], node[3]), moveDownUCS(node[2], node[3]), moveRightUCS(node[2], node[3]),
                 moveLeftUCS(node[2], node[3])]
    global nodesExpanded
    nodesExpanded += 1
    return expansion


# takes queue of nodes, list of nodes that were just expanded, and set of visited states
# if states in expansion have never been visited, enqueue them based on operator cost and return new queue
# may not be needed, due to aStar function
def uniformCost(nodes, expansion, visited):
    for state in expansion:
        if tuple(state[0]) not in visited:
            nodes.put((state[1], next(tie), state[0]))

    return nodes


# counts number of tiles not in the correct position, excluding blank (0)
# may not be needed, due to aStar function
def misplacedTile(nodes, expansion, visited):
    for state in expansion:
        numMisplaced = 0
        for i in range(len(state[0])):
            if (state[0])[i] != i + 1:
                if (state[0])[i] != 0:
                    numMisplaced += 1
        if tuple(state) not in visited:
            nodes.put((numMisplaced, next(tie), state[0]))

    return nodes


# counts number of tiles not in the correct position, excluding blank (0)
# returns number of misplaced tiles for aStar to use as heuristic
def countMisplacedTiles(p):
    numMisplaced = 0
    for i in range(len(p)):
        if p[i] != i + 1:
            if p[i] != 0:
                numMisplaced += 1
    return numMisplaced

# Helper Function
def manhattanDist(nodes, expansion, visited):
    for state in expansion:
        if tuple(state[0]) not in visited:
            manDist = sum(abs((val - 1) % 3 - i % 3) + abs((val - 1) // 3 - i // 3)
                          for i, val in enumerate(state[0]) if val)
            nodes.put((manDist, next(tie), state[0]))

    return nodes


# Sum of Manhattan Distance
def sumManhattanDistance(p):
    manDist = sum(abs((val - 1) % 3 - i % 3) + abs((val - 1) // 3 - i // 3)
                  for i, val in enumerate(p) if val)
    return manDist


# this search algorithm encompasses results
def aStar(nodes, expansion, visited, choice):
    # A* combines cost to reach this state with heuristic evaluation so g(n) + h(n)
    # with h(n) = 0 then it reduces to uniform cost search
    hn = 0
    # look at all new states and generate priority number based on user selected algorithms and enqueue node for that state
    # if user selects UCS then priority number is determined in operator functions
    for state in expansion:
        if tuple(state[0]) not in visited:
            gn = state[1]
            if choice == "2":
                hn = countMisplacedTiles(state[0])
            elif choice == "3":
                hn = sumManhattanDistance(state[0])
            nodes.put((gn + hn, next(tie), state[0], gn, hn))
            global maxNodesInQueue
            maxNodesInQueue = max(maxNodesInQueue, nodes.qsize())

    return nodes


def generalSearch(p, choice):
    nodes = queue.PriorityQueue()
    # priority_number, tie, puzzle, g(n), h(n)
    nodes.put((0, -1, p, 0, 0))
    global maxNodesInQueue
    maxNodesInQueue = max(maxNodesInQueue, nodes.qsize())
    print("Expanding state")
    # Takes care of duplicates
    visited = set()
    # Loop goes until goal is found or empty
    while True:
        # if queue ends up empty, the puzzle is unsolvable
        if nodes.empty():
            print("Failed -- puzzle unsolvable")
            return
        # grab node at head of queue
        node = nodes.get()
        print("The best state to expand with a g(n) = " + str(node[3]) + " and h(n) = " + str(node[4]) + " is...")
        printUCS(node[2])
        print("Expanding this node...\n")
        # mark this puzzle state as visited
        visited.add(tuple(node[2]))
        # if we've found goal state, return
        if isGoalState(node[2]):
            print("Goal!!")
            printUCS(node[2])
            global goalDepth
            goalDepth = node[3]
            return
        # if there's more nodes and we haven't found goal yet, expand curr node and enqueue using aStar algorithm
        nodes = aStar(nodes, expand(node), visited, choice)


def printUCS(puzzle):
    for i in range(len(puzzle)):
        if i % 3 == 0 and i > 0:
            print()
        if puzzle[i] == 0:
            print("0" + " ", end="")
        else:
            print(str(puzzle[i]) + " ", end='')
    print()


def userGeneratePuzzle():
    # prompt for user input
    print("Enter your puzzle, using zero to represent the blank and enter to represent end of a row")
    row1 = input("\nEnter 1st row, with space/tab between numbers\t")
    row2 = input("\nEnter 2nd row, with space/tab between numbers\t")
    row3 = input("\nEnter 3rd row, with space/tab between numbers\t")

    # concatenate all input strings
    p = row1 + " " + row2 + " " + row3
    # split concatenated string into a list and convert all entries to integers
    p = p.split()
    for i in range(len(p)):
        p[i] = int(p[i])

    return p

if __name__ == '__main__':
    main()