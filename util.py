import sys


class Field:
    '''
    A Field is the whole UTT board, both the macro and microboards.
    '''
    __EMPTY_FIELD = "."
    __AVAILABLE_FIELD = "-1"
    __NUM_COLS = 9
    __NUM_ROWS = 9
    __mBoard = []
    __mMacroboard = []
    __myId = 0
    __opponentId = 0

    # Initialize the macroboard and microboard with the empty field constant.
    def __init__(self):

        # Microboard
        null_row = []
        for col in range(self.__NUM_COLS):
            null_row.append(self.__EMPTY_FIELD)
        for row in range(self.__NUM_ROWS):
            self.__mBoard.append(list(null_row))

        # Macroboard
        null_row = []
        for col in range(self.__NUM_COLS // 3):
            null_row.append(self.__EMPTY_FIELD)
        for row in range(self.__NUM_ROWS//3):
            self.__mMacroboard.append(list(null_row))

    # Set microboard to string representation.
    # s - a string
    def parseFromString(self, s):
        s = s.replace(";", ",")
        r = s.split(",")

        counter = 0
        for y in range(self.__NUM_ROWS):
            for x in range(self.__NUM_COLS):
                self.__mBoard[x][y] = r[counter]
                counter += 1

    # Set macroboard to string representation.
    # s - a string
    def parseMacroboardFromString(self, s):
        r = s.split(",")
        counter = 0
        for y in range(3):
            for x in range(3):
                self.__mMacroboard[x][y] = r[counter]
                counter += 1

    # Return an array of all open spots on all the microboards.
    def getAvailableMoves(self):
        moves = []
        for y in range(self.__NUM_ROWS):
            for x in range(self.__NUM_COLS):
                if (self.isInActiveMicroboard(x, y) and (self.__mBoard[x][y] == self. __EMPTY_FIELD)):
                    moves.append(Move(x,y))
        return moves

    # Are (x, y) in a microboard that is still in play?
    # x, y - integers
    def isInActiveMicroboard(self, x, y):
        return self.__mMacroboard[x // 3][y // 3] == self. __AVAILABLE_FIELD

    # Returns a string representation of the field.
    def toString(self):
        r = ""
        counter = 0
        for y in range(self.__NUM_ROWS):
            for x in range(self.__NUM_COLS):
                if (counter > 0):
                    r += ","
                r += self.__mBoard[x][y]
                counter += 1
        return r

    # Are all microboards full?
    def isFull(self):
        for y in range(self.__NUM_ROWS):
            for x in range(self.__NUM_COLS):
                if (self.__mBoard[x][y] == self.__EMPTY_FIELD):  return False
        return True

    # Are all microboards empty?
    def isEmpty(self):
        for y in range(self.__NUM_ROWS):
            for x in range(self.__NUM_COLS):
                if (self.__mBoard[x][y] != self.__EMPTY_FIELD):  return False
        return True

    # Return number of columns.
    def getNrColumns(self):
        return self.__NUM_COLS

    # Return number of rows.
    def getNrRows(self):
        return self.__NUM_ROWS

    # Return number at this location, will be 0 or 1,
    # depending on player.
    # x, y - ints
    def getPlayerID(self, x, y):
        return self.__mBoard[x][y]

    # Return 0 or 1, depending on which player it is.
    def getMyId(self):
        return self.__myId

    # Set player id to 0 or 1.
    # id - either 0 or 1
    def setMyId(self, id):
        self.__myId = id

    # Return opponent id, either 0 or 1.
    def getOpponentId(self):
        return self.__opponentId

    # Set opponent id to either 0 or 1.
    # id - either 0 or 1
    def setOpponentId(self, id):
        self.__opponentId = id
        
        
        
class Move:
    '''
    A Move represents a UTT move. Used to get strings from bots to
    an engine readable format.
    '''

    __x = -1
    __y = -1

    # Set up move location.
    # x, y - ints
    def __init__(self, x,y):
        self.__x = x
        self.__y = y

    # Get the x coordinate of move.
    def getX(self):
        return self.__x

    # Get the y coordinate of move.
    def getY(self):
        return self.__y

    # Returns the engine readable string to make a move on the field.
    def toString(self):
        return "place_move {} {}".format(self.__x, self.__y)

class Player:
    '''
    A Player is a container for the UTT player's name.
    '''

    __name = ""
    def __init__(self, name):
        self.__name = name


class BotState:
    __MAX_TIMEBANK = -1
    __TIME_PER_MOVE = -1

    __roundNumber = -1
    __moveNumber = -1

    __timebank = -1
    __myName = ""
    __players = {}

    __field = None

    def __init__(self):
        self.__field = Field()
        self.__players = {}

    # Sets current time bot has in timebank.
    # value - int 
    def setTimebank(self, value):
        self.__timebank = value

    # Sets max time bot can have in timebank.
    # value - int
    def setMaxTimebank(self, value):
        self.__MAX_TIMEBANK = value

    # Sets time added to timebank each move.
    def setTimePerMove(self, value):
        self.__TIME_PER_MOVE = value

    # Sets bot's name.
    def setMyName(self, myName):
        self.__myName = myName

    # Sets current round number.
    # roundNumber - int
    def setRoundNumber(self, roundNumber):
        self.__roundNumber = roundNumber

    # Sets current move number.
    # moveNumber - int
    def setMoveNumber(self, moveNumber):
        self.__moveNumber = moveNumber

    # Returns current timebank.
    def getTimebank(self):
        return self.__timebank

    # Returns current round number.
    def getRoundNumber(self):
        return self.__roundNumber

    # Returns current move number.
    def getMoveNumber(self):
        return self.__moveNumber

    # Returns set of players
    def getPlayers(self):
        return self.__players

    # Returns current field.
    def getField(self):
        return self.__field

    # Returns bot's name.
    def getMyName(self):
        return self.__myName

    # Returns max possible timebank value.
    def getMaxTimebank(self):
        return self.__MAX_TIMEBANK

    # Returns time added to timebank each move.
    def getTimePerMove(self):
        return self.__TIME_PER_MOVE


class BotParser:
    '''
    A BotParser takes a bot and interprets its messages.
    '''
    __bot = None
    __currentState = None
    __log = None

    def __init__(self, bot):
        self.__bot = bot
        self.__currentState = BotState()
        self.__log = Log()

    # Prints out message and records to its log file that this is happening.
    def output(self, msg):
        self.__log.write("Sending: " + msg + " to stdout.")
        print(msg)
        sys.stdout.flush()

    # Main loop: reads input from bot and sends it to its message handler.
    def run(self):
        while not sys.stdin.closed:
            try:
                rawline = sys.stdin.readline()
                line = rawline.strip()
                self.handle_message(line)
            except EOFError:
                self.__log.write('EOF')
                self.__log.close()
        return

    # Parses a single command from a bot. Sends its command to the appropriate place.
    def handle_message(self, message):
        self.__log.write("bot received: {}\n".format(message))
        parts = message.split(" ")

        if not parts:
            self.__log.write("Unable to parse line (empty)\n")

        elif parts[0] == 'settings':
            self.parseSettings(parts[1], parts[2])

        elif parts[0] == 'update':
            if (parts[1] == "game"):
                self.parseGameData(parts[2], parts[3])

        elif parts[0] == 'action':
            if (parts[1] == "move"):
                if (len(parts) > 2):
                    self.__currentState.setTimebank(int(parts[2]))
                move = self.__bot.doMove(self.__currentState)

                if move != None:
                    #sys.stdout.write(move.toString())
                    self.output(move.toString())
                else:
                    #sys.stdout.write("pass")
                    self.output("pass")

        else:
            self.__log.write("Unknown command: {} \n".format(message))

    # Parse settings commands.
    def parseSettings(self, key, value):
        try:
            if key == "timebank":
                time = int(value)
                self.__currentState.setMaxTimebank(time)
                self.__currentState.setTimebank(time)

            elif key == "time_per_move":
                self.__currentState.setTimePerMove(int(value))

            elif key == "player_names":
                playerNames = value.split(",")
                for playerName in playerNames:
                    player = Player(playerName)
                    (self.__currentState.getPlayers())[playerName] =  player  # Check this

            elif key == "your_bot":
                self.__currentState.setMyName(value)

            elif key == "your_botid":
                myId = int(value)
                opponentId = 2 - myId + 1
                self.__currentState.getField().setMyId(myId)
                self.__currentState.getField().setOpponentId(opponentId)

            else:
                self.__log.write("Unable to parse settings input with key {}".format(key))

        except:
            self.__log.write("Unable to parse settings value {} for key {}".format(value, key))
            #e.printStackTrace()

    # Parse commands related to game data like the current round, the current move, or
    # the state of the game-field.
    def parseGameData(self, key, value):
        try:
            if key == "round":
                self.__currentState.setRoundNumber(int(value))

            elif key == "move":
                self.__currentState.setMoveNumber(int(value))

            elif key == "macroboard":
                self.__currentState.getField().parseMacroboardFromString(value);

            elif key == "field":
                self.__currentState.getField().parseFromString(value)

            else:
                self.__log.write("Cannot parse game data input with key {}".format(key))
        except:
            self.__log.write("Cannot parse game data value {} for key {}".format(value, key))
            #e.printStackTrace()


class Log:
    '''
    A Log encapsulates file writing. The logfiles it produces
    are records of how bots behaved.
    '''
    __FNAME = "/tmp/bot-log.txt"

    def __init__(self, fname=None):
        if (fname == None):
            import os

            pid = os.getpid()
            self.__FNAME = "/tmp/bot-log" + str(pid) + ".txt"
        else:
            self.__FNAME = fname

        self.__FILE = open(self.__FNAME, 'w')

    # Write to its file.
    def write(self, msg):
        self.__FILE.write(msg)

    # Close file output stream.
    def close(self):
        self.write("Closing log file.")
        self.__FILE.close()

