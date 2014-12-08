



import pygame




class Player1():


    x = None
    y = None
    speed = None
    padWid = None
    padHei = None
    score = None
    scoreFont = None


    def __init__(self, SCR_WID, SCR_HEI, SCREEN):

        self.screen = SCREEN

        self.screenWidth = SCR_WID
        self.screenHeight = SCR_HEI

        self.x = 16
        self.y = self.screenHeight/2

        self.speed = 3
        self.padWid, self.padHei = 8, 64
        self.score = 0
        self.scoreFont = pygame.font.Font("arial.ttf", 64)
    
    def scoring(self):
        scoreBlit = self.scoreFont.render(str(self.score), 1, (255, 255, 255))
        self.screen.blit(scoreBlit, (32, 16))
        if self.score == 10:
            print "Player 1 wins!"
            exit()
    
    def movement(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.y -= self.speed
        elif keys[pygame.K_s]:
            self.y += self.speed
    
        if self.y <= 0:
            self.y = 0
        elif self.y >= self.screenHeight-64:
            self.y = self.screenHeight-64
    
    def draw(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (self.x, self.y, self.padWid, self.padHei))








