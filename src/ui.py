# -*- coding: utf-8 -*-

import imgui
from imgui.integrations.pyglet import PygletRenderer

from pyglet.window import Window

from state import State
from settings import config

class UserInterface:
    def __init__(self, window: Window):
        imgui.create_context()
        self.window = window
        self.impl = PygletRenderer(window)
        self.font = self.load_font(config.mono_font_name, config.font_size)

    def load_font(self, font_path: str, font_size: int):
        io = imgui.get_io()

        win_w, win_h = self.window.width, self.window.height
        fb_w, fb_h = self.window.get_framebuffer_size()
        font_scaling_factor = max(float(fb_w) / win_w, float(fb_h) / win_h)

        new_font = io.fonts.add_font_from_file_ttf(font_path, font_size * font_scaling_factor )
        io.font_global_scale /= font_scaling_factor
        self.impl.refresh_font_texture()
        return new_font

    def clean_up(self):
        self.impl.shutdown()

    def update(self):
        imgui.new_frame()
        with imgui.font(self.font):
            self.draw()

    def render(self):
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def draw_menu_bar(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                    'Quit', 'Cmd+Q', False, True
                )
                if clicked_quit:
                    exit(1)
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def draw_car_positions(self):
        if State.cars == None:
            return;
        imgui.set_next_window_size(250, 240)
        imgui.set_next_window_position(20, 215)
        imgui.begin("Cars",
                    flags=imgui.WINDOW_NO_MOVE |
                    imgui.WINDOW_NO_TITLE_BAR  |
                    imgui.WINDOW_NO_RESIZE)
        headers = ['ID', 'Score', 'Mean Pos', 'Life']
        rows = []
        for car in State.cars:
            active = car['car'].awake
            rows.append((
                f"{car['id']}",
                f"{car['car'].position.x:.2f}",
                f"{car['mean_pos']:.2f}",
                f"{car['life']}"
            ))
        self.draw_table(headers, rows)
        imgui.end()

    def draw_table(self, headers, rows, table_name='DefTable'):
        imgui.columns(len(headers), table_name)
        for header in headers:
            imgui.text(header)
            imgui.next_column()
        imgui.separator()
        for row in rows:
            for item in row:
                imgui.text(item)
                imgui.next_column()
        imgui.columns(1)

    def draw_stats_box(self):
        imgui.set_next_window_size(200, 160)
        imgui.set_next_window_position(20, 40)
        imgui.begin("Simulation Stats",
                    flags=imgui.WINDOW_NO_MOVE |
                    imgui.WINDOW_NO_TITLE_BAR  |
                    imgui.WINDOW_NO_RESIZE)

        self.draw_table(headers=['Property', 'Value'],
                        rows=[
                            ('FPS', f'{State.fps:.2f}'),
                            ('Bodies', f'{State.num_bodies}'),
                            ('Joints', f'{State.num_joints}'),
                            ('Contacts', f'{State.num_contacts}'),
                            ('Proxies', f'{State.num_proxies}'),
                            ('Generation', f'{State.generation}')
                        ])
        imgui.end()

    def draw_sim_controls(self):
        imgui.begin("Simulation Controls")

        _, State.draw_aabb = imgui.checkbox("Draw AABBs", State.draw_aabb)
        _, State.draw_joints = imgui.checkbox("Draw Joints", State.draw_joints)
        _, State.draw_pairs = imgui.checkbox("Draw Pairs", State.draw_pairs)

        imgui.begin_group()
        imgui.text("First group (buttons):")
        imgui.button("Reset")
        imgui.button("Pause")
        imgui.end_group()

        imgui.end()

    def draw(self):
        self.draw_menu_bar()
        self.draw_stats_box()
        self.draw_sim_controls()
        self.draw_car_positions()
        imgui.show_test_window()
       
    def __del__(self):
        self.clean_up()
