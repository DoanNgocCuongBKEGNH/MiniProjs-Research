package game.gui;

import javax.swing.*;
import javax.swing.JOptionPane;
import java.awt.*;

import game.controls.*;

public class MainWindow extends JFrame{
	private GameCanvas myGC;
	private MainMenu myMenu;
	private HelpPage myHelp;
	private int mode;
	private Game myGame;
	private CardLayout c;
	//private JMenuBar myMB;
	
	public MainWindow(Game g) {
		this.myGame = g;
		this.mode = 1;
		myGC = new GameCanvas(this.myGame, this);
		myMenu = new MainMenu(this);
		myHelp = new HelpPage(this);
		c = new CardLayout();
		setLayout(c);
		
		add("Menu", myMenu);
		add("GC", myGC);
		add("Help", myHelp);
		
		setVisible(true);
		setSize(800, 600);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setTitle("Ô Ăn Quan");
		setLocationRelativeTo(null);
	}
	
	public void playGame() {
		this.myGame.playGame();
	}
	
	public void newGame() {
		this.myGame.restart();
		c.show(getContentPane(), "GC");
		mode = 2;
	}
	
	public void showHelp() {
		c.show(getContentPane(), "Help");
		mode = 3;
	}
	
	public void backToMain() {
		c.show(getContentPane(), "Menu");
		mode = 4;
	}
	
	public void exitGame() {
		int confirmed = JOptionPane.showConfirmDialog(null, "Are you sure you want to quit?", "Exit messge", JOptionPane.YES_NO_OPTION);
		if(confirmed == JOptionPane.YES_OPTION) {
			System.exit(0);
		}
	}
	
	public void redraw() {
		this.myGC.repaint();
	}
	
	public int getMode() {
		return mode;
	}
}
