# 틱택토 게임 Python 3.x

import random

def inputPlayerLetter():
    letter = ' '
    while not (letter=='X' or letter=='O') :         
        print ('Do you want be X or O?')
        letter = input().upper()

    # 튜플의 첫 번째 요소가 플레이어의 글자이고 두 번째 요소는 컴퓨터의 글자
    if letter == 'X':
        return 'X', 'O'
    else:
        return 'O', 'X'
    
def whoGoesFirst():
    if random.randint(0,1) : 
        return 'computer'
    else:
        return 'player'

def drawBoard(board):
    print( '+---+---+---+')
    print( '| '+board[7]+' | '+board[8]+' | '+board[9]+' |')
    print( '+---+---+---+')
    print( '| '+board[4]+' | '+board[5]+' | '+board[6]+' |')
    print( '+---+---+---+')
    print( '| '+board[1]+' | '+board[2]+' | '+board[3]+' |')
    print( '+---+---+---+')

def isSpaceFree(board, move):
    return board[move] ==' '

def isBoardFull(board):
    for i in range(1,10) :
        if board[i] ==' ' :  
             return False
    return True







# 2-3. 보드에 마크를 표시한다

def makeMove(board, letter, move):
    board[move] = letter

def isWinner(bo, le):
    # 보드(bo)와 플레이어 글자(le)를 파라미터로 받아,
    # (le)마크가 이겼을 때 True를 반환한다.
    return ((bo[7]==le and bo[8]==le and bo[9]==le) or
            (bo[4]==le and bo[5]==le and bo[6]==le) or
            (bo[1]==le and bo[2]==le and bo[3]==le) or
            (bo[7]==le and bo[4]==le and bo[1]==le) or
            (bo[8]==le and bo[5]==le and bo[2]==le) or
            (bo[9]==le and bo[6]==le and bo[3]==le) or
            (bo[7]==le and bo[5]==le and bo[3]==le) or
            (bo[9]==le and bo[5]==le and bo[1]==le))


# 더 똑똑한 컴퓨터를 만든다~

def undoMove(board, move):
    board[move]=' '
    

def getPlayerMove(board):
    move = 0
    while move not in range(1,10) or not isSpaceFree(board, move):
        move = input('What is your next move? (1-9)')
        if move.isdigit():
            move =  int(move)
        else:
            move = 0        
    return move

def getWinMove(board, letter):
    for move in range(1,10):
        if isSpaceFree(board, move):
            makeMove(board, letter, move)
            winResult=isWinner(board, letter)
            undoMove(board,move)
            if winResult:
                return move

    else: return 0


            

def getComputerMove(board, computerLetter):
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    move = getWinMove(board, computerLetter)
    if move in range(1,10):
        return move

    move = getWinMove(board, playerLetter)
    if move in range(1,10):
        return move




    
    loc1 = [7, 9, 1, 3]
    loc2 = [8, 4, 2, 6]
    random.shuffle(loc1)
    random.shuffle(loc2)
    locList = [5] + loc1 + loc2
    for i in locList :	
        if isSpaceFree(board, i):
            return i
             


playerLetter, computerLetter = inputPlayerLetter()
turn = whoGoesFirst()
print(turn, '가 먼저 시작합니다.')
theBoard= [' '] * 10
gameIsPlaying = True
while  gameIsPlaying:
    if turn == 'computer':
        move = getComputerMove(theBoard, computerLetter)
        makeMove(theBoard, computerLetter, move)
        if isWinner(theBoard, computerLetter ):
            drawBoard(theBoard)
            print('컴퓨터 승리..!')
            gameIsPlaying = False

        elif isBoardFull(theBoard):
            drawBoard(theBoard)
            print('The game is a tie!')
            gameIsPlaying =False

        else:
            turn = 'player'
    else:
        drawBoard(theBoard)
        move = getPlayerMove(theBoard)
        makeMove( theBoard, playerLetter, move)
        if isWinner(theBoard, playerLetter ):
            drawBoard(theBoard)
            print('만세!! 당신이 이겼네요..!')
            gameIsPlaying = False

        elif isBoardFull(theBoard):
            drawBoard(theBoard)
            print('The game is a tie!')
            gameIsPlaying =False

            

        else:
            turn = 'computer'
        
        
    


        
        
