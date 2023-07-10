package game.gui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.lang.Math;
import java.util.*;

import game.controls.Game;
import game.controls.board.*;

public class BoardDrawer extends Drawer {
	private Board myBoard;
	private Game myGame;
	private int cellSize = 100;
	private int danRadius = 5;
	private int quanRadius = 15;
	
	public BoardDrawer(Board myBoard, JPanel cp) {
		super(cp);
		this.myBoard = myBoard;
	}
	
	public BoardDrawer(Board myBoard, JPanel cp, int size, int dan, int quan) {
		super(cp);
		
		this.myBoard = myBoard;
		
		this.cellSize = size;
		this.danRadius = dan;
		this.quanRadius = quan;
	}
	
	public BoardDrawer(Game myGame, JPanel cp) {
		super(cp);
		this.myGame = myGame;
		this.myBoard = myGame.getBoard();
	}
	
	public BoardDrawer(Game myGame, JPanel cp, int size, int dan, int quan) {
		super(cp);
		this.myGame = myGame;
		this.myBoard = myGame.getBoard();
		
		this.cellSize = size;
		this.danRadius = dan;
		this.quanRadius = quan;
	}
	
	public void draw() {
		Graphics2D graphic2d = (Graphics2D) this.myG;
		setup(graphic2d);
		drawBoardLines(graphic2d);
		drawStones(graphic2d);
		if(this.myGame.isValidMove(onCellControl())) {
			highlightCell(graphic2d);
		}
		//graphic2d.drawOval(mouseX, mouseY, quanRadius, quanRadius);
	}
	
	public void setup(Graphics2D g) {
		g.clearRect(0, 0, width, height);
		g.setStroke(new BasicStroke(5));
		g.setFont(new Font("SansSerif", Font.BOLD, 20));
	}
	
	private void drawBoardLines(Graphics2D g) {
		g.setColor(Color.BLACK);
		for(int i = 0; i<10; i++) {
			int x = getWidth()/2 - (5*this.cellSize)/2 + (i%5)*this.cellSize;
			int y = getHeight()/2 - this.cellSize + (i/5)*this.cellSize;
			g.drawRect(x, y, this.cellSize, this.cellSize);
		}
		g.drawArc((getWidth()/2-(7*this.cellSize)/2), (getHeight()/2-this.cellSize), 2*this.cellSize, 2*this.cellSize, 90, 180);
		g.drawArc((getWidth()/2+(3*this.cellSize)/2), (getHeight()/2-this.cellSize), 2*this.cellSize, 2*this.cellSize, -90, 180);
	}
	
	private void drawStones(Graphics2D g) {
		BoardCell[] cells = this.myBoard.getCells();
		for(int i = 0; i<12; i++) {
			if(!cells[i].isOQuan()) {
				drawInDan(g, i, cells[i]);
			}
			else {
				drawInQuan(g, i, cells[i]);
			}
		}
	}
	
	private void drawInDan(Graphics2D g, int ind, BoardCell cell) {
		g.setColor(Color.BLACK);
		ArrayList<Stone> stones = cell.getStonesInCell();
		int ord = (ind<5)?ind:(5+(9-(ind-1)));
		int num = stones.size();
		int side = (int)Math.ceil(Math.sqrt(num));
		int d = (int)(this.cellSize*0.9/(side+1));
		int centerX = getWidth()/2 - 2*this.cellSize + (ord%5)*this.cellSize;
		int centerY = getHeight()/2 - this.cellSize/2 + (ord/5)*this.cellSize;
		for(int i=0; i<stones.size(); i++) {
			Stone s = stones.get(i);
			int r = this.danRadius;
			if(s.isQuan()) {
				r = this.quanRadius;
			}
			g.drawOval((int)(centerX + ((i%side)-(side*0.5-0.5))*d) - r/2, (int)(centerY - ((i/side)-(side*0.5-0.5))*d) - r/2, r, r);
		}
		g.setColor(new Color(150, 150, 150));
		g.drawString(cell.getPoint()+"", centerX - this.cellSize*9/20, centerY + this.cellSize*9/20);
	}
	
	private void drawInQuan(Graphics2D g, int ind, BoardCell cell) {
		g.setColor(Color.BLACK);
		ArrayList<Stone> stones = cell.getStonesInCell();
		int ord = (ind > 5) ? (-1):(1);
		int num = stones.size();
		int side = (int)Math.ceil(Math.sqrt(num/1.5));
		int d = (int)(this.cellSize*0.8/(side+1));
		int centerX = (int)(getWidth()/2 + ord*(2.5*this.cellSize + 0.4*this.cellSize));
		int centerY = getHeight()/2;
		for(int i=0; i<num; i++) {
			Stone s = stones.get(i);
			int r = this.danRadius;
			if(s.isQuan()) {
				r = this.quanRadius;
			}
			g.drawOval((int)(centerX - ord*((i%side)-(side*0.5-0.5))*d) - r/2, (int)(centerY - ((i/side)-(side*0.75-0.5))*d) - r/2, r, r);
		}
		g.setColor(new Color(150, 150, 150));
		g.drawString(cell.getPoint()+"", centerX - ord*this.cellSize/5 - (cell.getPoint()+"").length()*5, centerY + ord*this.cellSize*8/10 + 10);
	}
	
	public int onCell() {
		int x = (this.mouseX - width/2 + (5*this.cellSize)/2)/this.cellSize;
		int y = (this.mouseY - height/2 + this.cellSize)/this.cellSize;
		x = (x>=0 && x<5 && (this.mouseX - width/2 + (5*this.cellSize)/2)>0)?x:(-1);
		y = (y>=0 && y<2 && (this.mouseY - height/2 + this.cellSize)>0)?y:(-1);
		int i = (x>=0 && y>=0)?(x+y*5):(-1);
		return i;
	}
	
	public int onCellControl() {
		int ind = onCell();
		return (ind<5)?ind:(5+(9-(ind-1)));
	}
	
	public int getDir() {
		int ret = 1;
		int ind = onCell();
		int centerX = getWidth()/2 - 2*this.cellSize + (ind%5)*this.cellSize;
		if(mouseX>centerX == (ind>=0 && ind<5)) {
			ret = 1;
		}
		else{
			ret = -1;
		}
		return ret;
	}
	
	public void highlightCell(Graphics2D g) {
		if(this.myGame.isValidMove(onCellControl())) {
			int i = onCell();
			if(i>=0) {
				int x = getWidth()/2 - (5*this.cellSize)/2 + (i%5)*this.cellSize;
				int y = getHeight()/2 - this.cellSize + (i/5)*this.cellSize;
				int cx = x + this.cellSize/2;
				int cy = y + this.cellSize/2;
				Polygon leftArrow = new Polygon(new int[] {x+10, x+10, x-10}, new int[] {cy-10, cy+10, cy}, 3);
				Polygon rightArrow = new Polygon(new int[] {x+cellSize-10, x+cellSize-10, x+cellSize+10}, new int[] {cy-10, cy+10, cy}, 3);
				g.setColor(Color.BLUE);
				g.drawRect(x, y, this.cellSize, this.cellSize);
				g.setColor(Color.GREEN);
				if(mouseX>cx) {
					g.drawPolygon(rightArrow);
					g.fillPolygon(rightArrow);
				}
				else {
					g.drawPolygon(leftArrow);
					g.fillPolygon(leftArrow);
				}
			}
		}
	}
	
	public void mouseClicked(MouseEvent e) {
		this.myGame.setMove(onCellControl(), getDir());
	}
}
