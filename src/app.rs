use std::time::{Duration, Instant};

use egui::{
    emath::Rot2, vec2, Align2, Color32, DragValue, FontId, Key, Rounding, Sense, Shape, Stroke,
    Vec2, Widget,
};
use ndarray::Array2;

use crate::{
    game::{Cell, CellState, Field},
    solver::{predict, Prediction},
};

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct Minesweeper {
    #[serde(skip)] // This how you opt-out of serialization of a field
    field: Field,
    #[serde(skip)]
    game_over: bool,
    new_field_size: (usize, usize),
    new_field_mines: usize,
    #[serde(skip)]
    selected: Option<(usize, usize)>,
    #[serde(skip)]
    predictions: Option<Array2<Option<Prediction>>>,
    #[serde(skip)]
    last_predictions_time: Option<Duration>,
}

impl Default for Minesweeper {
    fn default() -> Self {
        let new_field_size = (25, 25);
        let new_field_mines = 40;
        Self {
            field: Field::new(new_field_size, new_field_mines)
                .expect("initializing field using fixed values"),
            game_over: false,
            new_field_size,
            new_field_mines,
            selected: None,
            predictions: None,
            last_predictions_time: None,
        }
    }
}

impl Minesweeper {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for Minesweeper {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_dark_light_mode_switch(ui);
            });
        });

        egui::SidePanel::left("left_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Board size:");
                ui.add(DragValue::new(&mut self.new_field_size.0));
                ui.label("by");
                ui.add(DragValue::new(&mut self.new_field_size.1));
            });
            ui.horizontal(|ui| {
                ui.label("Mines:");
                ui.add(DragValue::new(&mut self.new_field_mines));
            });
            if ui.button("New Game").clicked() {
                if let Some(field) = Field::new(self.new_field_size, self.new_field_mines) {
                    self.field = field;
                    self.game_over = false;
                    if let Some(predictions) = self.predictions.as_mut() {
                        let (preds, t) = self.field.get_predictions();
                        *predictions = preds;
                        self.last_predictions_time = Some(t);
                    }
                }
            }
            if ui
                .checkbox(&mut self.predictions.is_some(), "Show Predictions")
                .clicked()
            {
                if self.predictions.is_none() {
                    let (preds, t) =  self.field.get_predictions();
                    self.predictions = Some(preds);
                    self.last_predictions_time = Some(t);
                } else {
                    self.predictions = None;
                }
            };
            if let Some(t) = self.last_predictions_time {
                ui.label(format!("Last predictions time: {t:?}"));
            }
            ui.label(format!("Remaining mines: {}", self.field.remaining_mines()));
        });

        let mut flagged = Vec::new();
        let mut cleared = Vec::new();
        let game_complete = self.field.complete();

        {
            let ([up, down, left, right], [space, flag, esc, restart]) = ctx.input(|inp| {
                (
                    [
                        [Key::ArrowUp, Key::W, Key::K],
                        [Key::ArrowDown, Key::S, Key::J],
                        [Key::ArrowLeft, Key::A, Key::H],
                        [Key::ArrowRight, Key::D, Key::L],
                    ]
                    .map(|keys| keys.into_iter().any(|key| inp.key_pressed(key))),
                    [Key::Space, Key::F, Key::Escape, Key::R].map(|key| inp.key_pressed(key)),
                )
            });

            if restart {
                if let Some(field) = Field::new(self.new_field_size, self.new_field_mines) {
                    self.field = field;
                    self.game_over = false;
                    if let Some(predictions) = self.predictions.as_mut() {
                        let (preds, t) = self.field.get_predictions();
                        *predictions = preds;
                        self.last_predictions_time = Some(t);
                    }
                }
            } else if let Some((x, y)) = self.selected.as_mut() {
                let (width, height) = self.field.size();
                if esc {
                    self.selected = None;
                } else if space {
                    cleared.push((*x, *y));
                } else if flag {
                    flagged.push((*x, *y));
                } else {
                    if up && *y > 0 {
                        *y -= 1;
                    }
                    if down && *y < height - 1 {
                        *y += 1;
                    }
                    if left && *x > 0 {
                        *x -= 1;
                    }
                    if right && *x < width - 1 {
                        *x += 1;
                    }
                }
            } else if up || down || left || right || space || flag {
                self.selected = Some((0, 0));
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let grid_size = ui.available_size();
            let field_size = self.field.size();
            let cell_size = f32::min(
                grid_size.x / field_size.0 as f32,
                grid_size.y / field_size.1 as f32,
            );
            let final_grid_size = vec2(field_size.0 as f32, field_size.1 as f32) * cell_size;

            ui.scope(|ui| {
                ui.spacing_mut().interact_size = Vec2::ZERO;
                let response = egui::Grid::new("field").spacing((0.0, 0.0)).show(ui, |ui| {
                    for (y, row) in self
                        .field
                        .board
                        .lanes(ndarray::Axis(0))
                        .into_iter()
                        .enumerate()
                    {
                        for (x, cell) in row.indexed_iter() {
                            let response = ui.add(
                                cell.show(
                                    cell_size,
                                    self.selected == Some((x, y)),
                                    self.predictions
                                        .as_ref()
                                        .and_then(|predictions| predictions[(x, y)]),
                                ),
                            );
                            if !self.game_over && !game_complete {
                                if response.clicked() {
                                    if ctx.input(|inp| inp.modifiers.shift) {
                                        flagged.push((x, y));
                                    } else {
                                        cleared.push((x, y));
                                    }
                                }
                                if response.secondary_clicked() {
                                    flagged.push((x, y));
                                }
                            }
                        }
                        ui.end_row();
                    }
                });
                if self.game_over {
                    ui.painter().text(
                        response.response.rect.center(),
                        Align2::CENTER_CENTER,
                        "GAME\nOVER",
                        FontId::proportional(final_grid_size.x.min(final_grid_size.y) / 4.0),
                        Color32::RED,
                    );
                }
                if game_complete {
                    ui.painter().text(
                        response.response.rect.center(),
                        Align2::CENTER_CENTER,
                        "YOU\nWIN",
                        FontId::proportional(final_grid_size.x.min(final_grid_size.y) / 4.0),
                        Color32::GREEN,
                    );
                }
            });
        });

        let board_changed = !flagged.is_empty() || !cleared.is_empty();

        for pos in flagged {
            self.field.toggle_flag(pos);
        }
        for pos in cleared {
            // Try clearing the cell, if that is invalid, try clearing its neighbors
            if self
                .field
                .clear_cell(pos)
                .or_else(|| self.field.clear_neighbors(pos))
                .unwrap_or_default()
            {
                self.game_over = true;
            }
        }

        if board_changed {
            if let Some(predictions) = self.predictions.as_mut() {
                let (preds, t) = self.field.get_predictions();
                *predictions = preds;
                self.last_predictions_time = Some(t);
            }
        }
    }
}

impl Field {
    fn get_predictions(&self) -> (Array2<Option<Prediction>>, Duration) {
        let t0 = Instant::now();
        let predictions =
            predict(self).mapv_into_any(|pred| pred.map(Prediction::from_probability));
        let t1 = Instant::now();
        (predictions, t1 - t0)
    }
}

impl Cell {
    fn show(self, size: f32, selected: bool, prediction: Option<Prediction>) -> CellWidget {
        CellWidget {
            cell: self,
            size,
            selected,
            prediction,
        }
    }
}

struct CellWidget {
    cell: Cell,
    size: f32,
    selected: bool,
    prediction: Option<Prediction>,
}

impl Widget for CellWidget {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(self.size), Sense::click());

        let stroke = Stroke::from((rect.width() / 16.0, ui.style().visuals.strong_text_color()));

        let painter = ui.painter();

        painter.rect(
            rect.shrink(stroke.width / 2.0),
            Rounding::ZERO,
            ui.style().visuals.window_fill(),
            (
                stroke.width,
                if self.selected {
                    Color32::BLUE
                } else {
                    ui.style().visuals.window_stroke().color
                },
            ),
        );

        let draw_inner_border = || {
            painter.rect_stroke(
                rect.shrink(stroke.width * 1.5),
                Rounding::ZERO,
                (
                    stroke.width,
                    match (self.selected, response.hovered()) {
                        (true, false) => Color32::BLUE,
                        (_, true) => Color32::LIGHT_BLUE,
                        _ => stroke.color,
                    },
                ),
            );
        };

        match self.cell.state {
            CellState::Unrevealed => {
                draw_inner_border();

                if let Some(prediction) = self.prediction {
                    let color = match prediction {
                        Prediction::Free => Color32::GREEN,
                        Prediction::Mine => Color32::RED,
                        Prediction::Contradiction => Color32::LIGHT_RED,
                        Prediction::Probability(prob, _) => Color32::YELLOW.gamma_multiply(prob),
                    };
                    painter.rect_filled(rect.shrink(rect.height() / 5.0), Rounding::ZERO, color);
                }
            }
            CellState::Flagged => {
                draw_inner_border();
                let flag_rect = rect.shrink2(rect.size() / vec2(3.0, 4.0));
                painter.line_segment([flag_rect.left_top(), flag_rect.left_bottom()], stroke);
                painter.rect_filled(
                    flag_rect.with_max_y(flag_rect.center().y),
                    Rounding::ZERO,
                    stroke.color,
                );
            }
            CellState::Revealed => {
                painter.text(
                    rect.center() + vec2(0.0, rect.height() / 20.0),
                    Align2::CENTER_CENTER,
                    self.cell.neighbors.to_string(),
                    FontId::monospace(rect.height() * 0.8),
                    stroke.color,
                );
            }
            CellState::Exploded => {
                let draw_starburst = |radius, color| {
                    let outer_rad = vec2(0.0, radius);
                    let inner_rad = outer_rad * 0.75;

                    let points = (0..=16)
                        .map(|r| {
                            let v = if r % 2 == 0 { inner_rad } else { outer_rad };
                            let rot = Rot2::from_angle(r as f32 / -16.0 * std::f32::consts::TAU);
                            rect.center() + (rot * v)
                        })
                        .collect();
                    painter.add(Shape::convex_polygon(points, color, Stroke::NONE));
                };

                draw_starburst(rect.height() / 2.5, Color32::YELLOW);
                draw_starburst(rect.height() / 4.0, Color32::LIGHT_RED);
            }
            CellState::Empty => {}
        }

        response
    }
}
