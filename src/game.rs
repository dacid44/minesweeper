use ndarray::Array2;
use rand::{distributions::Uniform, rngs::SmallRng, Rng, SeedableRng};

#[derive(Debug)]
pub(crate) struct Field {
    pub(crate) board: Array2<Cell>,
    mines: usize,
    rng: SmallRng,
    is_new: bool,
}

impl Field {
    pub(crate) fn size(&self) -> (usize, usize) {
        self.board.dim()
    }

    pub(crate) fn complete(&self) -> bool {
        // The game is complete when we have no more unrevealed (or flagged) spaces that are not mines
        !self.board.iter().any(|cell| {
            matches!(cell.state, CellState::Unrevealed | CellState::Flagged) && !cell.mine
        })
    }

    /// Returns the number of total mines minus the number of total flags
    pub(crate) fn remaining_mines(&self) -> usize {
        let mines = self.board.iter().filter(|cell| cell.mine).count();
        let flags = self
            .board
            .iter()
            .filter(|cell| cell.state == CellState::Flagged)
            .count();
        mines.saturating_sub(flags)
    }

    /// Returns None if either dimension was zero, or too many mines were specified than can (reasonably)
    /// fit on the board.
    pub(crate) fn new(size: (usize, usize), mines: usize) -> Option<Self> {
        if size.0 == 0 || size.1 == 0 || mines > (size.0 * size.1 + 1) / 2 {
            return None;
        }

        let board = Array2::<Cell>::default(size);
        let rng = SmallRng::from_entropy();
        let mut field = Self {
            board,
            mines,
            rng,
            is_new: true,
        };

        field.init_board();
        Some(field)
    }

    pub(crate) fn clear(&mut self) {
        self.is_new = true;
        self.board.fill(Default::default());
        self.init_board();
    }

    fn init_board(&mut self) {
        let x_d = Uniform::new(0, self.size().0);
        let y_d = Uniform::new(0, self.size().1);

        let mut placed_mines = 0;
        while placed_mines < self.mines {
            let mine_pos = (self.rng.sample(x_d), self.rng.sample(y_d));
            if self.board[mine_pos].mine {
                continue;
            }

            self.board[mine_pos].mine = true;

            for neighbor in neighbors(&self.board, mine_pos) {
                self.board[neighbor].neighbors += 1;
            }

            placed_mines += 1;
        }
    }

    /// Returns a bool signifying if a mine has exploded. Returns None if the given cell has already
    /// been cleared or flagged, or if the given cell is invalid.
    pub(crate) fn clear_cell(&mut self, pos: (usize, usize)) -> Option<bool> {
        match (self.is_new, self.board.get_mut(pos)?.reveal()?) {
            (_, RevealStatus::Empty) => {}
            (true, _) => {
                self.clear();
                return self.clear_cell(pos);
            }
            (false, RevealStatus::Exploded) => return Some(true),
            (false, RevealStatus::Safe) => return Some(false),
        }

        self.is_new = false;

        // If the cell was empty, clear neighboring empty cells
        let mut check = neighbors(&self.board, pos).collect::<Vec<_>>();

        while let Some(next_pos) = check.pop() {
            if matches!(self.board[next_pos].reveal(), Some(RevealStatus::Empty)) {
                check.extend(neighbors(&self.board, next_pos));
            }
        }

        Some(false)
    }

    /// Returns a bool signifying that the flag was valid (i.e., that the cell was not already
    /// revealed). Returns None if the cell was invalid.
    pub(crate) fn toggle_flag(&mut self, pos: (usize, usize)) -> Option<bool> {
        self.board.get_mut(pos).map(|cell| {
            self.is_new = false;
            cell.toggle_flag()
        })
    }

    pub(crate) fn clear_neighbors(&mut self, pos: (usize, usize)) -> Option<bool> {
        let cell = self.board.get(pos)?;
        if cell.state != CellState::Revealed
            || neighbors(&self.board, pos)
                .filter(|pos| self.board[*pos].state == CellState::Flagged)
                .count()
                != cell.neighbors as usize
        {
            return None;
        }

        let mut exploded = false;
        for pos in neighbors(&self.board, pos) {
            if self.clear_cell(pos).unwrap_or_default() {
                exploded = true;
            }
        }

        Some(exploded)
    }
}

pub(crate) fn neighbors<T>(
    board: &Array2<T>,
    (x, y): (usize, usize),
) -> impl Iterator<Item = (usize, usize)> {
    [
        (x.wrapping_sub(1), y.wrapping_sub(1)),
        (x, y.wrapping_sub(1)),
        (x + 1, y.wrapping_sub(1)),
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1),
        (x.wrapping_sub(1), y + 1),
        (x.wrapping_sub(1), y),
    ]
    .map(|pos| board.get(pos).map(|_| pos))
    .into_iter()
    .flatten()
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Cell {
    pub(crate) state: CellState,
    pub(crate) neighbors: u8,
    mine: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            state: CellState::Unrevealed,
            neighbors: 0,
            mine: false,
        }
    }
}

impl Cell {
    /// Returns None if the cell has already been cleared or flagged.
    fn reveal(&mut self) -> Option<RevealStatus> {
        match self.state {
            CellState::Unrevealed if self.mine => {
                self.state = CellState::Exploded;
                Some(RevealStatus::Exploded)
            }
            CellState::Unrevealed if self.neighbors == 0 => {
                self.state = CellState::Empty;
                Some(RevealStatus::Empty)
            }
            CellState::Unrevealed => {
                self.state = CellState::Revealed;
                Some(RevealStatus::Safe)
            }
            _ => None,
        }
    }

    /// Returns a bool signifying that the flag was valid (i.e., that the cell was not already
    /// revealed).
    fn toggle_flag(&mut self) -> bool {
        match self.state {
            CellState::Unrevealed => {
                self.state = CellState::Flagged;
                true
            }
            CellState::Flagged => {
                self.state = CellState::Unrevealed;
                true
            }
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CellState {
    /// Initial state
    Unrevealed,
    /// Flagged
    Flagged,
    /// Clicked on, showing a number
    Revealed,
    /// Clicked on, was a mine
    Exploded,
    /// Clicked on, no mines
    Empty,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RevealStatus {
    Exploded,
    Safe,
    Empty,
}
