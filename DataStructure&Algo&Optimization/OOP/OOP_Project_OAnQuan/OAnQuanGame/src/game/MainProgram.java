package game;

import game.controls.Game;
import game.gui.MainWindow;

public class MainProgram {
	public static void main(String[] args) {
		Game myGame = new Game();
		MainWindow myWindow = new MainWindow(myGame);
		
		while(true) {
			while(myWindow.getMode() == 2) {
				myWindow.redraw();
				myWindow.playGame();
			}
			for(int i=0; i<5000; i++) {}
		}
	}
}
