



import pygame




class Ball():


    def __init__(self, SCR_WID, SCR_HEI, PLAYER1, PLAYER2, SCREEN):

        self.screen = SCREEN
        
        self.player1 = PLAYER1
        self.player2 = PLAYER2

        self.screenWidth = SCR_WID
        self.screenHeight = SCR_HEI

        self.x = self.screenWidth/2
        self.y = self.screenHeight/2

        self.speed_x = -3
        self.speed_y = 3
        self.size = 8



    def movement(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # wall col
        if self.y <= 0:
            self.speed_y *= -1
        elif self.y >= self.screenHeight-self.size:
            self.speed_y *= -1

        if self.x <= 0:
            self.__init__(self.screenWidth, self.screenHeight, self.player1, self.player2, self.screen)
            self.player2.score += 1
        elif self.x >= self.screenWidth-self.size:
            self.__init__(self.screenWidth, self.screenHeight, self.player1, self.player2, self.screen)
            self.speed_x = 3
            self.player1.score += 1
        # wall col
        # paddle col
        # self.player1
        for n in range(-self.size, self.player1.padHei):
            if self.y == self.player1.y + n:
                if self.x <= self.player1.x + self.player1.padWid:
                    self.speed_x *= -1
                    break
            n += 1
        # player2
        for n in range(-self.size, self.player2.padHei):
            if self.y == self.player2.y + n:
                if self.x >= self.player2.x - self.player2.padWid:
                    self.speed_x *= -1
                    break
            n += 1
        # paddle col



    def draw(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (self.x, self.y, 8, 8))







