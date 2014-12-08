



# imports
import pygame
from PongPlayer1 import Player1 as P1
from PongPlayer2 import Player2 as P2
from PongBall import Ball





SCR_WID = 640
SCR_HEI = 480 

screen = pygame.display.set_mode((SCR_WID, SCR_HEI))
pygame.display.set_caption("Pong")
pygame.font.init()
clock = pygame.time.Clock()
FPS = 60


player1 = P1(SCR_WID, SCR_HEI, screen)
player2 = P2(SCR_WID, SCR_HEI, screen)
ball = Ball(SCR_WID, SCR_HEI, player1, player2, screen)


while True:
	#process
	for event in pygame.event.get():
			if event.type == pygame.QUIT:
				print "Game exited by user"
				exit()
	##process
	#logic
	ball.movement()
	player1.movement()
	player2.movement()
	##logic
	#draw
	screen.fill((0, 0, 0))
	ball.draw()
	player1.draw()
	player1.scoring()
	player2.draw()
	player2.scoring()
	##draw
	#_______
	pygame.display.flip()
	clock.tick(FPS)
