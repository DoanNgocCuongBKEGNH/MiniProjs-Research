package game.gui;

import java.awt.*;
import javax.swing.*;

import game.controls.*;
import game.controls.player.Player;

public class PlayerDrawer extends Drawer{
	private Game myGame;
	private int d = 150;
	
	public PlayerDrawer(Game myGame, JPanel cp) {
		super(cp);
		
		this.myGame = myGame;
	}
	
	public void draw() {
		Graphics2D g2d = (Graphics2D) this.myG; 
		setup(g2d);
		drawPlayers(g2d);
	}
	
	public void setup(Graphics2D g) {
		g.setStroke(new BasicStroke(5));
		g.setColor(Color.BLACK);
		g.setFont(new Font("SansSerif", Font.BOLD, 30));
	}
	
	public void drawPlayers(Graphics2D g) {
		FontMetrics metrics = g.getFontMetrics(g.getFont());
		String p1 = this.myGame.getPlayer1().getPoint()+"";
		String p2 = this.myGame.getPlayer2().getPoint()+"";
		
		g.drawRect(width/2 - d/2, 0, d, d*4/5);
		g.drawString("Player 2", width/2 - metrics.stringWidth("Player 2")/2, d/5+metrics.getHeight()/4);
		g.drawString(p1, width/2 - metrics.stringWidth(p1)/2, d/2+metrics.getHeight()/4);
		g.drawRect(width/2 - d/2, height - d*4/5, d, d*4/5);
		g.drawString("Player 1", width/2 - metrics.stringWidth("Player 1")/2, height-d/5+metrics.getHeight()/4);
		g.drawString(p2, width/2 - metrics.stringWidth(p2)/2, height-d/2+metrics.getHeight()/4);
		
		g.setColor(Color.MAGENTA);
		g.setFont(new Font("SansSerif", Font.BOLD, 20));
		if(!this.myGame.gameEnd()) {
			if(this.myGame.isP1Turn()) {
				g.drawRect(width/2 - d/2, 0, d, d*4/5);
				g.drawString("Player 2's turn", d/2, d/5);
			}
			else {
				g.drawRect(width/2 - d/2, height - d*4/5, d, d*4/5);
				g.drawString("Player 1's turn", d/2, d/5);
			}
		}
		else {
			int win = 3 - this.myGame.getWinner();
			if(win>0) {
				g.drawString("Player " + win + " win", d/2, d/5);
			}
			else {
				g.drawString("Tie", d/2, d/5);
			}
		}
		
	}
}
