from manim import *


class AttentionGeometricWarping(Scene):
    def construct(self):
        # ==========================================
        # 1. SCENE SETUP: Vector Space Coordinate Grid
        # ==========================================
        grid = NumberPlane(
            x_range=[-1, 6, 1],
            y_range=[-1, 5, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            },
        )
        self.play(Create(grid), run_time=1.5)

        # Title Label
        title = Text("Attention Geometric Warping", font_size=28).to_edge(UP)
        self.play(Write(title))

        # ==========================================
        # 2. DEFINE INITIAL GEOMETRIC POINTS
        # ==========================================
        # Coordinates in the vector space
        origin = grid.c2p(0, 0)
        pos_bank_static = grid.c2p(1, 3)
        pos_muddy = grid.c2p(5, 1)
        pos_river = grid.c2p(4, 4)

        # Intermediate step: Dynamic target vector closer to 'muddy'
        # Blended vector inside the convex hull
        pos_bank_contextual = grid.c2p(4.2, 1.4)

        # Create Static Word Vectors (Arrows)
        v_bank = Arrow(origin, pos_bank_static, buff=0, color=YELLOW)
        v_muddy = Arrow(origin, pos_muddy, buff=0, color=RED)
        v_river = Arrow(origin, pos_river, buff=0, color=BLUE)

        # Labels for the word nodes
        lbl_bank = Text("bank (static)", font_size=16, color=YELLOW).next_to(
            pos_bank_static, UP
        )
        lbl_muddy = Text("muddy (Value)", font_size=16, color=RED).next_to(
            pos_muddy, DR
        )
        lbl_river = Text("river (Value)", font_size=16, color=BLUE).next_to(
            pos_river, UR
        )

        # Render initial layout
        self.play(
            GrowArrow(v_bank),
            Write(lbl_bank),
            GrowArrow(v_muddy),
            Write(lbl_muddy),
            GrowArrow(v_river),
            Write(lbl_river),
            run_time=2,
        )
        self.wait(1)

        # ==========================================
        # 3. CONVEX HULL REGION ENCLOSURE
        # ==========================================
        # The space spanned by the context coordinates (Values)
        # Bounded between origin, river, and muddy
        convex_hull = Polygon(
            origin,
            pos_river,
            pos_muddy,
            stroke_color=PURPLE,
            stroke_width=2,
            fill_color=PURPLE,
            fill_opacity=0.15,
        )

        hull_label = Text(
            "Context Value Space (Convex Hull)", font_size=16, color=PURPLE
        )
        hull_label.move_to(grid.c2p(2.5, 1.8))

        # Explain the area
        self.play(Create(convex_hull), FadeIn(hull_label))
        self.wait(1.5)

        # ==========================================
        # 4. ANIMATE THE ATTENTION DRAG / WARP
        # ==========================================
        # Highlight alignment connection (Query to Key)
        alignment_beam = Line(
            pos_bank_static, pos_muddy, stroke_color=WHITE, stroke_width=3, color=WHITE
        )
        align_lbl = Text("High Q-K Alignment Score", font_size=14, color=WHITE).next_to(
            alignment_beam.get_center(), UR
        )

        self.play(Create(alignment_beam), Write(align_lbl))
        self.play(alignment_beam.animate.set_opacity(0.3), FadeOut(align_lbl))

        # Prepare contextual label changes
        lbl_bank_contextual = Text(
            "bank (contextual)", font_size=16, color=GREEN
        ).next_to(pos_bank_contextual, UR)

        # Simultaneously morph the static vector into its new shifted spatial coordinate position
        self.play(
            Transform(v_bank, Arrow(origin, pos_bank_contextual, buff=0, color=GREEN)),
            Transform(lbl_bank, lbl_bank_contextual),
            run_time=3.5,
            rate_func=exponential_decay,
        )
        self.wait(2)

        # Clean exit text
        summary = Text(
            "Landscape deformed: Meaning pulled into context boundary.",
            font_size=18,
            color=GREEN_A,
        ).to_edge(DOWN)
        self.play(Write(summary))
        self.wait(3)
