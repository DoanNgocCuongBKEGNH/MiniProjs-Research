package game.controls.board;

import java.util.*;

public class BoardCell {
	private ArrayList<Stone> stonesInCell;
	private boolean isOQuan;
	
	public BoardCell(boolean isOQuan) {
		stonesInCell = new ArrayList<Stone>();
		this.isOQuan = isOQuan;
		if(isOQuan) {
			this.stonesInCell.add(new Stone(true));
		}
		else {
			for(int i=0; i<5; i++) {
				this.stonesInCell.add(new Stone(false));
			}
		}
	}
	
	public ArrayList<Stone> getStonesInCell() {
		return this.stonesInCell;
	}
	
	public int getNumberOfStones() {
		return this.stonesInCell.size();
	}
	
	public int getPoint() {
		int point = 0;
		for(Stone s: this.stonesInCell) {
			point += s.getValue();
		}
		return point;
	}
	
	public boolean isOQuan() {
		return this.isOQuan;
	}
	
}
