
import curses

import sys
sys.path.append(f"{sys.path[0]}/..")

from lib.Engine.engine_classes import *

# initialization
screen = curses.initscr()
win = curses.newwin(20 + 2, 60 + 2, 0, 20)

# config settings
curses.noecho()
curses.cbreak()
curses.curs_set(False)
win.keypad(True)

# window setting
win.box()

if __name__ == "__main__":
	
	# initialize game
	vs = VectorSpace([Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)])
	cs =  CoordinateSystem(vs, Point(0, 0, 0))
	es = EventSystem()
	elip = HyperEllipsoid(cs, Point(20, 0, 0), Vector(1, 0, 0), [6, 8, 10])
	plane = HyperPlane(cs, Point(40, 0, 0), Vector(1.0, 0.5, 0.0))
	# ent_list = EntitiesList([plane, elip])
	ent_list = EntitiesList([elip])
	camera = Camera(cs, Point(0, 0, 0), Vector(10, 0, 0), math.pi / 2, 60, math.pi / 2)
	G = Game(cs, es, ent_list)
	GC = Canvas(cs, ent_list, 20, 60)
	GC.update(camera)

	# init event system
	G.es.add("OnCameraBodyMove")
	G.es.add("OnCameraHeadMove")
	G.es.add_handle("OnCameraBodyMove", camera_body_move_handler)
	G.es.add_handle("OnCameraHeadMove", camera_head_move_handler)


	while True:
		lines = GC.draw_curse()
		char = win.getch()
		
		if char == ord('q'):
			curses.nocbreak()
			win.keypad(False)
			curses.echo()
			curses.endwin()
			break

		elif char == ord('p'):
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])

		# body move
		elif char == curses.KEY_RIGHT:
			win.addstr(0, 0, 'right')
			G.es.trigger("OnCameraBodyMove", camera, curses.KEY_RIGHT)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])

		elif char == curses.KEY_LEFT:
			win.addstr(0, 0, 'left ')
			G.es.trigger("OnCameraBodyMove", camera, curses.KEY_LEFT)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])
			   
		elif char == curses.KEY_UP:
			win.addstr(0, 0, 'up   ')
			G.es.trigger("OnCameraBodyMove", camera, curses.KEY_UP)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])

		elif char == curses.KEY_DOWN:
			win.addstr(0, 0, 'down ')
			G.es.trigger("OnCameraBodyMove", camera, curses.KEY_DOWN)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])
		
		# head move
		elif char == ord('a'):
			win.addstr(0, 0, 'turnL')
			G.es.trigger("OnCameraHeadMove", camera, char)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])
		
		elif char == ord('d'):
			win.addstr(0, 0, 'turnR')
			G.es.trigger("OnCameraHeadMove", camera, char)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])
		
		elif char == ord('w'):
			win.addstr(0, 0, 'turnU')
			G.es.trigger("OnCameraHeadMove", camera, char)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])
		
		elif char == ord('s'):
			win.addstr(0, 0, 'turnD')
			G.es.trigger("OnCameraHeadMove", camera, char)
			GC.update(camera)
			lines = GC.draw_curse()
			for i in range(len(lines)):
				win.move(i + 1, 1)
				win.addstr(lines[i])
	
	