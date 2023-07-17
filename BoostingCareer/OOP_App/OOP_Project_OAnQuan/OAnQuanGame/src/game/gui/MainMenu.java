package game.gui;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class MainMenu extends JPanel{
	private JButton start;
	private JButton help;
	private JButton exit;
	private JLabel title;
	private MainWindow pa;
	
	public MainMenu(MainWindow pa) {
		this.pa = pa;
		title = new JLabel("Ô Ăn Quan", JLabel.CENTER);
		title.setFont(new Font("SansSerif", Font.PLAIN, 50));
		
		start = new JButton("New game");
		help = new JButton("Help");
		exit = new JButton("Exit");
		
		ActionListener startGame = new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				pa.newGame();
			}
		};
		
		ActionListener openHelp = new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				pa.showHelp();
			}
		};
		
		ActionListener exitGame = new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				pa.exitGame();
			}
		};
		
		start.addActionListener(startGame);
		help.addActionListener(openHelp);
		exit.addActionListener(exitGame);
		
		setLayout(new GridBagLayout());
		GridBagConstraints c = new GridBagConstraints();
		
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 0;
		c.gridy = 0;
		c.gridwidth = 4;
		c.ipady = 50;
		add(title, c);
		
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 0;
		c.gridy = 1;
		c.gridwidth = 1;
		c.ipady = 0;
		add(start, c);
		
		c.gridx = 1;
		c.gridy = 1;
		add(help, c);
		
		c.gridx = 2;
		c.gridy = 1;
		add(exit, c);
	}
}
