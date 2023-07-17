package game.controls.board;

public class Board {
	private BoardCell[] cells;
	
	public Board() {
		cells = new BoardCell[12];
		for(int i = 0; i<12; i++) {
			boolean quan = (i == 5) || (i==11);
			this.cells[i] = new BoardCell(quan);
		}
	}
	
	public void reset() {
		cells = new BoardCell[12];
		for(int i = 0; i<12; i++) {
			boolean quan = (i == 5) || (i==11);
			this.cells[i] = new BoardCell(quan);
		}
	}
	
	public BoardCell[] getCells() {
		return this.cells;
	}
	
	public boolean gameEnd() {
		return (cells[5].getNumberOfStones() == 0 && cells[11].getNumberOfStones() == 0);
	}
}
