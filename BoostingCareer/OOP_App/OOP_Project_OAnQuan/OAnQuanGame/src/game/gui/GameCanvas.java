package game.gui;

import game.controls.board.*;
import game.controls.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class GameCanvas extends JPanel{
	private BoardDrawer myBD;
	private PlayerDrawer myPD;
	private JButton back;
	private MainWindow pa;
	
	public GameCanvas(Board myBoard) {
		this.myBD = new BoardDrawer(myBoard, this);
	}
	public GameCanvas(Game myGame, MainWindow pa) {
		this.pa = pa;
		this.myBD = new BoardDrawer(myGame, this);
		this.myPD = new PlayerDrawer(myGame, this);
		
		ActionListener goBack = new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				int confirmed = JOptionPane.showConfirmDialog(null, "Are you sure you want to quit?", "Confirm quit", JOptionPane.YES_NO_OPTION);
				if (confirmed == JOptionPane.YES_OPTION) {
					pa.backToMain();
				}
			}
		};
		
		this.back = new JButton("Back");
		this.back.setLocation(getWidth() - this.back.getWidth(), getHeight()-this.back.getHeight());
		this.back.addActionListener(goBack);
		
		add(this.back);
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		
		((Graphics2D)g).setBackground(new Color(242, 242, 242));
		
		this.back.setLocation(getWidth() - this.back.getWidth(), getHeight()-this.back.getHeight());
		
		this.myBD.setMyG(g);
		this.myPD.setMyG(g);
		
		this.myBD.draw();
		this.myPD.draw();
	}
}
