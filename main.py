import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY
import numpy as np
import random
import time
import sys
import os
from collections import deque
import urllib.request # 画像ダウンロード用に追加
import urllib.error   # エラーハンドリング用に追加

# --- アクションセットの定義 ---
ACTION_SET_COMPLEX = COMPLEX_MOVEMENT
ACTION_SET_SIMPLE = SIMPLE_MOVEMENT
ACTION_SET_RIGHT_ONLY = RIGHT_ONLY

RIGHT_ONLY_NO_JUMP = [
    ['NOOP'],
    ['right'],
    ['right', 'B'],
]
# ACTION_SET_RIGHT_ONLY_NO_JUMP = RIGHT_ONLY_NO_JUMP

WALK_NO_JUMP = [
    ['NOOP'],
    ['right'],
    ['left'],
]
# ACTION_SET_WALK_ONLY_NO_JUMP = WALK_NO_JUMP
RIGHT_DASH_ONLY_NO_JUMP = [
    ['right', 'B'],
]
# ACTION_SET_RIGHT_DASH_ONLY_NO_JUMP = RIGHT_DASH_ONLY_NO_JUMP
RIGHT_WALK_ONLY_NO_JUMP = [
    ['right'],
]

RIGHT_DASH_JUMP_ONLY = [
    ['right', 'A', 'B'],
]
# ACTION_SET_RIGHT_DASH_JUMP_ONLY = RIGHT_DASH_JUMP_ONLY

# --- X座標によるアクションセット切り替え設定 ---
X_THRESHOLDS_FOR_ACTION_SET_SWITCH = [0]
ALLOWED_ACTIONS_SUBSETS_BY_X = [
    ACTION_SET_RIGHT_ONLY,
    # ACTION_SET_SIMPLE,
]
ACTION_SET_NAMES_BY_X = [
    "RIGHT_ONLY",
    # "SIMPLE",
]

# 7-4
# X_THRESHOLDS_FOR_ACTION_SET_SWITCH = [0, 435, 815, 900, 1050, 1150, 1520, 1755, 2260, 2400]
# ALLOWED_ACTIONS_SUBSETS_BY_X = [
#     ACTION_SET_RIGHT_ONLY,
#     RIGHT_DASH_ONLY_NO_JUMP,
#     RIGHT_DASH_JUMP_ONLY,
#     RIGHT_DASH_ONLY_NO_JUMP,
#     RIGHT_DASH_JUMP_ONLY,
#     RIGHT_ONLY_NO_JUMP,
#     RIGHT_DASH_JUMP_ONLY,
#     ACTION_SET_RIGHT_ONLY,
#     RIGHT_ONLY_NO_JUMP,
#     ACTION_SET_RIGHT_ONLY,
# ]
# ACTION_SET_NAMES_BY_X = [
#     "RIGHT_ONLY",
#     "RIGHT_DASH_ONLY_NO_JUMP",
#     "RIGHT_DASH_JUMP_ONLY",
#     "RIGHT_DASH_ONLY_NO_JUMP",
#     "RIGHT_DASH_JUMP_ONLY",
#     "RIGHT_ONLY_NO_JUMP",
#     "RIGHT_DASH_JUMP_ONLY",
#     "RIGHT_ONLY",
#     "RIGHT_ONLY_NO_JUMP",
#     "RIGHT_ONLY",
# ]

# 6-2
# X_THRESHOLDS_FOR_ACTION_SET_SWITCH = [0, 1370, 1400]
# ALLOWED_ACTIONS_SUBSETS_BY_X = [
#     ACTION_SET_RIGHT_ONLY,
#     ACTION_SET_SIMPLE,
#     ACTION_SET_RIGHT_ONLY,
# ]
# ACTION_SET_NAMES_BY_X = [
#     "RIGHT_ONLY",
#     "SIMPLE",
#     "RIGHT_ONLY",
# ]

# 4-4
# X_THRESHOLDS_FOR_ACTION_SET_SWITCH = [0, 915, 1050, 1450, 1550, 1675]
# ALLOWED_ACTIONS_SUBSETS_BY_X = [
#     ACTION_SET_RIGHT_ONLY,
#     ACTION_SET_RIGHT_ONLY_NO_JUMP,
#     ACTION_SET_RIGHT_ONLY,
#     ACTION_SET_SIMPLE,
#     ACTION_SET_WALK_ONLY_NO_JUMP,
#     ACTION_SET_RIGHT_ONLY,
  
# ]
# ACTION_SET_NAMES_BY_X = [
#     "RIGHT_ONLY",
#     "RIGHT_ONLY_NO_JUMP",
#     "RIGHT_ONLY",
#     "SIMPLE", 
#     "WALK_ONLY_NO_JUMP",
#     "RIGHT_ONLY",
# ]

# X_THRESHOLDS_FOR_ACTION_SET_SWITCH = [0, 500, 1500]
# ALLOWED_ACTIONS_SUBSETS_BY_X = [
#     ACTION_SET_SIMPLE,
#     ACTION_SET_COMPLEX,
#     ACTION_SET_RIGHT_ONLY
# ]
# ACTION_SET_NAMES_BY_X = [
#     "SIMPLE (X < 500)",
#     "COMPLEX (500 <= X < 1500)",
#     "RIGHT_ONLY (X >= 1500)"
# ]

# 次元の確認
if len(X_THRESHOLDS_FOR_ACTION_SET_SWITCH) != len(ALLOWED_ACTIONS_SUBSETS_BY_X):
    raise ValueError(f"X_THRESHOLDS_FOR_ACTION_SET_SWITCH ({len(X_THRESHOLDS_FOR_ACTION_SET_SWITCH)}) と ALLOWED_ACTIONS_SUBSETS_BY_X ({len(ALLOWED_ACTIONS_SUBSETS_BY_X)}) の長さが一致していません")

if len(X_THRESHOLDS_FOR_ACTION_SET_SWITCH) != len(ACTION_SET_NAMES_BY_X):
    raise ValueError(f"X_THRESHOLDS_FOR_ACTION_SET_SWITCH ({len(X_THRESHOLDS_FOR_ACTION_SET_SWITCH)}) と ACTION_SET_NAMES_BY_X ({len(ACTION_SET_NAMES_BY_X)}) の長さが一致していません")

current_action_set_config_idx = 0

INITIAL_ACTION_SET_FOR_ENV = ACTION_SET_COMPLEX # JoypadSpace はこれで初期化
INITIAL_ACTION_SET_NAME_FOR_ENV = "COMPLEX_BASE" # (デバッグ用)

CURRENT_ACTION_SET = ALLOWED_ACTIONS_SUBSETS_BY_X[current_action_set_config_idx] # 探索時のフィルタリング用
CURRENT_ACTION_SET_NAME = ACTION_SET_NAMES_BY_X[current_action_set_config_idx]   # 表示用

# --- Pygame 定数 (変更なし) ---
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 600
GAME_SCREEN_BASE_WIDTH = 256
GAME_SCREEN_BASE_HEIGHT = 240
GAME_SCALE_FACTOR = 2
GAME_SCREEN_WIDTH = GAME_SCREEN_BASE_WIDTH * GAME_SCALE_FACTOR
GAME_SCREEN_HEIGHT = GAME_SCREEN_BASE_HEIGHT * GAME_SCALE_FACTOR
GAME_SCREEN_X = 30
GAME_SCREEN_Y = 50
GAME_SCREEN_RECT = pygame.Rect(GAME_SCREEN_X, GAME_SCREEN_Y, GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT)
CONTROLLER_DISPLAY_WIDTH = 360
CONTROLLER_X = GAME_SCREEN_X + GAME_SCREEN_WIDTH + 30
BACKGROUND_COLOR = (30, 30, 30)
HIGHLIGHT_COLOR = pygame.Color("yellow")
FPS = 120
ACTION_INTERVAL = 1/12

# --- コントローラー画像とボタン定義 (変更なし) ---
CONTROLLER_IMAGE_PATH = 'fig/famicon01_01.png'
DPAD_UP_RECT_ORIG = pygame.Rect(208 - 46, 340 + 22, 30, 40)
DPAD_DOWN_RECT_ORIG = pygame.Rect(208 - 46, 436 + 22, 30, 40)
DPAD_LEFT_RECT_ORIG = pygame.Rect(155 - 46, 393 + 22, 40, 30)
DPAD_RIGHT_RECT_ORIG = pygame.Rect(250 - 46, 393 + 22, 40, 30)
BUTTON_B_CENTER_ORIG = (549, 461)
BUTTON_B_RADIUS_ORIG = 28
BUTTON_A_CENTER_ORIG = (644, 461)
BUTTON_A_RADIUS_ORIG = 28
ORIGINAL_IMG_WIDTH_FOR_COORDS = 800
ORIGINAL_IMG_HEIGHT_FOR_COORDS = 800
scaled_button_geometries = {}
COMMAND_TO_KEY = {
    'right': 'dpad_right', 'left': 'dpad_left', 'up': 'dpad_up', 'down': 'dpad_down',
    'A': 'button_a', 'B': 'button_b', 'NOOP': 'noop'
}
BUTTON_GEOMETRIES_ORIG = {
    'dpad_up': {'type': 'rect', 'geom': DPAD_UP_RECT_ORIG},
    'dpad_down': {'type': 'rect', 'geom': DPAD_DOWN_RECT_ORIG},
    'dpad_left': {'type': 'rect', 'geom': DPAD_LEFT_RECT_ORIG},
    'dpad_right': {'type': 'rect', 'geom': DPAD_RIGHT_RECT_ORIG},
    'button_a': {'type': 'circle', 'geom': (BUTTON_A_CENTER_ORIG, BUTTON_A_RADIUS_ORIG)},
    'button_b': {'type': 'circle', 'geom': (BUTTON_B_CENTER_ORIG, BUTTON_B_RADIUS_ORIG)},
}

# --- フィードバック機構のためのパラメータ (変更なし) ---
# MAX_SUCCESSFUL_SEQUENCES = 1
MIN_X_PROGRESS_FOR_SUCCESS = 10
SHORT_TERM_FAILURE_MEMORY_SIZE = 0
A_BUTTON_MAX_HOLD_TIME = 1.8
THRESHOLD_NORMAL = 60
THRESHOLD_AFTER_FALL = 90

# --- グローバル変数 (変更なし/追加あり) ---
screen = None
clock = None
controller_base_image_scaled = None
font_small = None
font_medium = None
CONTROLLER_RECT = None
CONTROLLER_Y = None

successful_sequences = []
short_term_failure_actions = deque(maxlen=SHORT_TERM_FAILURE_MEMORY_SIZE)
overall_best_x_pos = 40
a_button_press_time_start = 0
info = {}
use_fall_threshold_next_episode = False
previous_x_pos = 40
g_x_pos_at_loop_warp = -1
g_special_replay_control_active = False

def update_pygame_caption():
    global CURRENT_ACTION_SET_NAME # CURRENT_ACTION_SET_NAME を使う
    pygame.display.set_caption(f"Super Mario Bros - AI ({CURRENT_ACTION_SET_NAME})")


def init_pygame():
    global screen, clock, controller_base_image_scaled, font_small, font_medium
    global scaled_button_geometries, CONTROLLER_RECT, CONTROLLER_X, CONTROLLER_Y
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    update_pygame_caption()
    clock = pygame.time.Clock()
    font_small = pygame.font.Font(None, 28)
    font_medium = pygame.font.Font(None, 36)

    # --- コントローラー画像のダウンロード処理 ---
    controller_image_url = 'https://stockmaterial.net/wp/wp-content/uploads/img/famicon01_01.png'
    controller_image_dir = os.path.dirname(CONTROLLER_IMAGE_PATH) # 'fig'

    if not os.path.exists(CONTROLLER_IMAGE_PATH):
        print(f"コントローラー画像 '{CONTROLLER_IMAGE_PATH}' が見つかりません。")
        print(f"'{controller_image_url}' からダウンロードを試みます...")
        
        try:
            # ディレクトリ 'fig' が存在しない場合は作成
            if controller_image_dir and not os.path.exists(controller_image_dir):
                os.makedirs(controller_image_dir)
                print(f"ディレクトリ '{controller_image_dir}' を作成しました。")

            # urllib.request を使ってダウンロード
            urllib.request.urlretrieve(controller_image_url, CONTROLLER_IMAGE_PATH)
            print(f"画像を '{CONTROLLER_IMAGE_PATH}' にダウンロードしました。")
        except urllib.error.URLError as e:
            print(f"エラー: 画像のダウンロードに失敗しました (URLエラー)。URLやネットワーク接続を確認してください: {e}")
            print(f"URL: {controller_image_url}")
            print("手動で画像をダウンロードして 'fig/famicon01_01.png' として保存してください。")
            pygame.quit()
            sys.exit()
        except OSError as e:
            print(f"エラー: 画像の保存に失敗しました (OSエラー)。ディレクトリの権限などを確認してください: {e}")
            print(f"パス: {CONTROLLER_IMAGE_PATH}")
            print("手動で画像をダウンロードして 'fig/famicon01_01.png' として保存してください。")
            pygame.quit()
            sys.exit()
        except Exception as e: # その他の予期せぬエラー
            print(f"エラー: 画像のダウンロード中に予期せぬエラーが発生しました: {e}")
            print("手動で画像をダウンロードして 'fig/famicon01_01.png' として保存してください。")
            pygame.quit()
            sys.exit()

    # --- 画像の読み込み (既存の処理の続き) ---
    # ダウンロード試行後に再度ファイルの存在を確認
    if not os.path.exists(CONTROLLER_IMAGE_PATH):
        print(f"エラー: コントローラー画像 '{CONTROLLER_IMAGE_PATH}' がダウンロード後も見つかりません。処理を中断します。")
        pygame.quit()
        sys.exit()

    try:
        controller_base_image_orig = pygame.image.load(CONTROLLER_IMAGE_PATH).convert_alpha()
    except pygame.error as e:
        print(f"エラー: コントローラー画像 '{CONTROLLER_IMAGE_PATH}' を読み込めませんでした: {e}")
        pygame.quit(); sys.exit()

    original_controller_img_width = controller_base_image_orig.get_width()
    original_controller_img_height = controller_base_image_orig.get_height()
    if original_controller_img_width == 0:
        print(f"エラー: コントローラー画像 '{CONTROLLER_IMAGE_PATH}' の幅が0です。")
        pygame.quit(); sys.exit()
    
    aspect_ratio = original_controller_img_height / original_controller_img_width
    calculated_controller_display_height = int(CONTROLLER_DISPLAY_WIDTH * aspect_ratio)

    controller_base_image_scaled = pygame.transform.smoothscale(
        controller_base_image_orig, (CONTROLLER_DISPLAY_WIDTH, calculated_controller_display_height)
    )
    CONTROLLER_Y = GAME_SCREEN_Y + (GAME_SCREEN_HEIGHT - calculated_controller_display_height) // 2
    CONTROLLER_RECT = pygame.Rect(CONTROLLER_X, CONTROLLER_Y, CONTROLLER_DISPLAY_WIDTH, calculated_controller_display_height)

    scale_x = CONTROLLER_DISPLAY_WIDTH / ORIGINAL_IMG_WIDTH_FOR_COORDS
    scale_y = calculated_controller_display_height / ORIGINAL_IMG_HEIGHT_FOR_COORDS 

    for key, data in BUTTON_GEOMETRIES_ORIG.items():
        if data['type'] == 'rect':
            orig_rect = data['geom']
            scaled_rect = pygame.Rect(
                orig_rect.left * scale_x, orig_rect.top * scale_y,
                max(1, int(orig_rect.width * scale_x)), max(1, int(orig_rect.height * scale_y))
            )
            scaled_button_geometries[key] = {'type': 'rect', 'geom': scaled_rect}
        elif data['type'] == 'circle':
            orig_center, orig_radius = data['geom']
            scaled_center_x = orig_center[0] * scale_x
            scaled_center_y = orig_center[1] * scale_y
            scaled_radius = max(1, int(orig_radius * min(scale_x, scale_y)))
            scaled_button_geometries[key] = {'type': 'circle', 'geom': ((scaled_center_x, scaled_center_y), scaled_radius)}

def init_mario_env():
    global INITIAL_ACTION_SET_FOR_ENV # ★★★ 初期化用のアクションセットを使用
    try:
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
    except Exception:
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array')
    env = JoypadSpace(env, INITIAL_ACTION_SET_FOR_ENV)
    return env

def convert_frame_to_pygame_surface(frame_np):
    frame_np_swapped = np.transpose(frame_np, (1, 0, 2))
    surface = pygame.surfarray.make_surface(frame_np_swapped)
    return surface

def draw_controller_state_ui(surface, base_image, pressed_button_keys_set, controller_screen_rect):
    if controller_screen_rect is None: return
    surface.blit(base_image, controller_screen_rect.topleft)
    for key_to_highlight in pressed_button_keys_set:
        if key_to_highlight in scaled_button_geometries:
            info_geom = scaled_button_geometries[key_to_highlight]
            if info_geom['type'] == 'rect':
                highlight_rect_on_screen = info_geom['geom'].move(controller_screen_rect.left, controller_screen_rect.top)
                pygame.draw.rect(surface, HIGHLIGHT_COLOR, highlight_rect_on_screen, border_radius=3)
            elif info_geom['type'] == 'circle':
                center_in_img_coords, radius = info_geom['geom']
                screen_center_x = center_in_img_coords[0] + controller_screen_rect.left
                screen_center_y = center_in_img_coords[1] + controller_screen_rect.top
                pygame.draw.circle(surface, HIGHLIGHT_COLOR, (screen_center_x, screen_center_y), radius)

def draw_text_info(surface, current_action_str_list_disp, total_reward, steps, episode_num, current_x_pos_info, is_replaying_disp, current_threshold):
    global font_small, CONTROLLER_RECT, CURRENT_ACTION_SET_NAME # ★★★ 表示用に CURRENT_ACTION_SET_NAME を使う
    if CONTROLLER_RECT is None: return

    mode_str = " (Replay)" if is_replaying_disp else f" (Explore - {CURRENT_ACTION_SET_NAME})" # ★★★ モード表示にアクションセット名追加
    action_display_str = "Action: " + " ".join(current_action_str_list_disp) + mode_str
    action_text_surf = font_small.render(action_display_str, True, (255, 255, 255))
    action_text_pos = (CONTROLLER_RECT.left, CONTROLLER_RECT.bottom + 15)
    surface.blit(action_text_surf, action_text_pos)

    stats_text_str = f"Ep: {episode_num} | Rew: {total_reward:.0f} | Steps: {steps} | X: {current_x_pos_info} | Thr: {current_threshold}"
    stats_text_surf = font_small.render(stats_text_str, True, (255, 255, 255))
    stats_text_pos = (GAME_SCREEN_RECT.left, GAME_SCREEN_RECT.bottom + 15)
    surface.blit(stats_text_surf, stats_text_pos)

    help_text_surf = font_small.render("R:Reset ESC:Quit", True, (200,200,200))
    surface.blit(help_text_surf, (SCREEN_WIDTH - help_text_surf.get_width() - 10, SCREEN_HEIGHT - help_text_surf.get_height() -10))


def update_memory_at_episode_end(final_info_dict, episode_frame_by_frame_actions_log, current_ep_max_x, episode_total_reward):
    global successful_sequences, short_term_failure_actions, overall_best_x_pos, use_fall_threshold_next_episode
    # (この関数は前回から変更なし、X座標付きログの保存と、ループ時の成功シーケンス非保存ロジックは維持)
    print(f"\n--- Updating memory based on episode outcome (Ep Max X: {current_ep_max_x})---")
    print(f"DEBUG update_memory_at_episode_end: Called. loop_detected_event: {final_info_dict.get('loop_detected_event', False)}, overall_best_x_pos before update: {overall_best_x_pos}, current_ep_max_x: {current_ep_max_x}")

    cleared = final_info_dict.get('flag_get', False)
    is_significant_progress = current_ep_max_x > overall_best_x_pos + MIN_X_PROGRESS_FOR_SUCCESS or \
                             (current_ep_max_x > overall_best_x_pos and cleared)

    if final_info_dict.get('loop_detected_event', False):
        print(f"  Loop detected event. is_significant_progress: {is_significant_progress} (based on overall_best_x_pos={overall_best_x_pos}). successful_sequences will not be updated with this loop-causing sequence.")
    elif cleared or is_significant_progress:
        print(f"  Episode successful (Cleared: {cleared}, Progress: {is_significant_progress}). Evaluating sequence of {len(episode_frame_by_frame_actions_log)} frames.")
        new_sequence_data = {
            "sequence_with_x": list(episode_frame_by_frame_actions_log),
            "max_x": current_ep_max_x,
            "score": episode_total_reward,
            "cleared": cleared
        }
        if not successful_sequences:
            successful_sequences.append(new_sequence_data)
            print(f"  New best sequence stored: X={new_sequence_data['max_x']}, Cleared={new_sequence_data['cleared']}")
        else:
            current_best_sequence = successful_sequences[0]
            new_is_better = (new_sequence_data["cleared"] and not current_best_sequence["cleared"]) or \
                             (new_sequence_data["cleared"] == current_best_sequence["cleared"] and new_sequence_data["max_x"] > current_best_sequence["max_x"]) or \
                             (new_sequence_data["cleared"] == current_best_sequence["cleared"] and new_sequence_data["max_x"] == current_best_sequence["max_x"] and new_sequence_data["score"] > current_best_sequence["score"])
            if new_is_better:
                print(f"  Found a new best sequence (Old: X={current_best_sequence['max_x']}, Clr={current_best_sequence['cleared']} | New: X={new_sequence_data['max_x']}, Clr={new_sequence_data['cleared']}). Replacing.")
                successful_sequences = [new_sequence_data]
            else:
                print(f"  Current episode's sequence is not better than the stored best (Stored: X={current_best_sequence['max_x']}, Clr={current_best_sequence['cleared']}). Not updating.")
        if successful_sequences:
             print(f"    Current best sequence: X={successful_sequences[0]['max_x']}, Cleared={successful_sequences[0]['cleared']}, Len={len(successful_sequences[0]['sequence_with_x'])} frames")

    if final_info_dict.get('loop_detected_event', False):
        print("  Reason for next threshold: Episode ended due to loop detection.")
        use_fall_threshold_next_episode = False
    elif cleared:
        print("  Reason for next threshold: Stage cleared.")
        use_fall_threshold_next_episode = False
    elif final_info_dict.get('time', 400) <= 1:
        print("  Reason for next threshold: Time up.")
        use_fall_threshold_next_episode = False
    else:
        y_pos_end = final_info_dict.get('y_pos', 0)
        player_state = final_info_dict.get('player_state', 0)
        is_fall_death = y_pos_end >= 250 or player_state == 0x0b
        if is_fall_death:
            print("  Reason for next threshold: Mario died from falling or similar cause.")
            use_fall_threshold_next_episode = True
        else:
             print("  Reason for next threshold: Mario died, but not from falling (and not time/clear/loop).")
             use_fall_threshold_next_episode = False

    is_timeout_for_short_term = final_info_dict.get('time', 400) <= 1
    if not cleared and not is_timeout_for_short_term and not final_info_dict.get('loop_detected_event', False):
        if episode_frame_by_frame_actions_log:
            last_executed_action_idx, _ = episode_frame_by_frame_actions_log[-1]
            if SHORT_TERM_FAILURE_MEMORY_SIZE > 0 and last_executed_action_idx not in short_term_failure_actions:
                short_term_failure_actions.append(last_executed_action_idx)
            # ★★★ 表示するアクション文字列は INITIAL_ACTION_SET_FOR_ENV から取得 ★★★
            print(f"  Last executed frame action '{INITIAL_ACTION_SET_FOR_ENV[last_executed_action_idx]}' considered for short-term failure memory (non-clear, non-timeout, non-loop failure).")
        elif SHORT_TERM_FAILURE_MEMORY_SIZE > 0:
            print("  Mario failed (non-clear, non-timeout, non-loop) with no actions logged for short-term failure memory.")

    if not final_info_dict.get('loop_detected_event', False):
        overall_best_x_pos = max(overall_best_x_pos, current_ep_max_x)
        print(f"  Updated overall_best_x_pos (non-loop event) to: {overall_best_x_pos}")
    else:
        print(f"  overall_best_x_pos ({overall_best_x_pos}) remains unchanged due to loop_detected_event (was set to loop warp point).")
    print(f"DEBUG update_memory_at_episode_end: overall_best_x_pos after update: {overall_best_x_pos}")


def game_loop(env):
    global screen, clock, controller_base_image_scaled, font_small, font_medium
    global successful_sequences, short_term_failure_actions, overall_best_x_pos, a_button_press_time_start
    global info, use_fall_threshold_next_episode, THRESHOLD_NORMAL, THRESHOLD_AFTER_FALL
    global previous_x_pos, g_x_pos_at_loop_warp, g_special_replay_control_active
    global current_action_set_config_idx, CURRENT_ACTION_SET, CURRENT_ACTION_SET_NAME, INITIAL_ACTION_SET_FOR_ENV # ★★★

    max_episodes = 10000
    for episode_count in range(1, max_episodes + 1):

        if use_fall_threshold_next_episode:
            current_threshold = THRESHOLD_AFTER_FALL
            print(f"Ep {episode_count}: Using FALL threshold ({current_threshold}) due to previous fall death.")
            use_fall_threshold_next_episode = False
        else:
            current_threshold = THRESHOLD_NORMAL

        state_frame, info = env.reset()
        previous_x_pos = info.get('x_pos', 40)
        current_episode_max_x = info.get('x_pos', 40) # ★★★ エピソード開始時に初期化

        # ★★★ エピソード開始時のアクションセット設定 (current_episode_max_x に基づいて再評価) ★★★
        # このタイミングで current_episode_max_x は初期X座標なので、通常は最初の設定になる
        initial_config_idx = 0
        for i in range(len(X_THRESHOLDS_FOR_ACTION_SET_SWITCH) -1, -1, -1):
            if current_episode_max_x >= X_THRESHOLDS_FOR_ACTION_SET_SWITCH[i]:
                initial_config_idx = i
                break
        if initial_config_idx != current_action_set_config_idx: # 通常は初回のみか、リセット後
            print(f"DEBUG Ep {episode_count} Start: Initializing action set config based on X={current_episode_max_x}.")
            current_action_set_config_idx = initial_config_idx
            CURRENT_ACTION_SET = ALLOWED_ACTIONS_SUBSETS_BY_X[current_action_set_config_idx]
            CURRENT_ACTION_SET_NAME = ACTION_SET_NAMES_BY_X[current_action_set_config_idx]
            update_pygame_caption()

        print(f"DEBUG Ep {episode_count} Start: current_action_set_config_idx: {current_action_set_config_idx}, Name: {CURRENT_ACTION_SET_NAME}")
        print(f"  g_special_replay_control_active: {g_special_replay_control_active}, g_x_pos_at_loop_warp: {g_x_pos_at_loop_warp}, overall_best_x_pos: {overall_best_x_pos}")
        if successful_sequences:
            print(f"  DEBUG Ep Start: successful_sequences[0]['max_x']: {successful_sequences[0]['max_x']}, len: {len(successful_sequences[0]['sequence_with_x'])}")


        apply_special_replay_exit_this_ep = False
        target_x_for_special_replay_exit = -1

        if g_special_replay_control_active:
            if g_x_pos_at_loop_warp >= 0 and overall_best_x_pos <= g_x_pos_at_loop_warp:
                apply_special_replay_exit_this_ep = True
                target_x_for_special_replay_exit = g_x_pos_at_loop_warp
                print(f"Ep {episode_count}: Special replay exit condition ACTIVE. Target X: {target_x_for_special_replay_exit}. current overall_best_x_pos: {overall_best_x_pos}")
            else:
                print(f"Ep {episode_count}: Deactivating special loop replay control. overall_best_x_pos ({overall_best_x_pos}) vs g_x_pos_at_loop_warp ({g_x_pos_at_loop_warp}).")
                g_special_replay_control_active = False
                g_x_pos_at_loop_warp = -1
        print(f"DEBUG Ep {episode_count} After special control check: apply_special_replay_exit_this_ep: {apply_special_replay_exit_this_ep}, target_x_for_special_replay_exit: {target_x_for_special_replay_exit}")

        done = False
        episode_frame_by_frame_log = deque()
        last_action_choice_time = time.time()
        total_episode_reward = 0
        episode_step_count = 0
        # current_episode_max_x は上で初期化済み

        is_replaying_sequence = False
        current_replay_actions_with_x = []
        replay_pointer = 0
        num_frames_to_replay_in_current_segment = 0

        action_idx_for_this_frame = -1 # INITIAL_ACTION_SET_FOR_ENV でのインデックス
        current_held_action_idx = -1   # INITIAL_ACTION_SET_FOR_ENV でのインデックス

        if successful_sequences:
            best_sequence_data = successful_sequences[0]
            full_actions_with_x = best_sequence_data["sequence_with_x"]
            num_actual_replay_frames = len(full_actions_with_x) - current_threshold
            if num_actual_replay_frames > 0:
                is_replaying_sequence = True
                current_replay_actions_with_x = full_actions_with_x
                replay_pointer = 0
                num_frames_to_replay_in_current_segment = num_actual_replay_frames
                print(f"Ep {episode_count}: Starting in REPLAY mode. Full seq len: {len(full_actions_with_x)}, Replaying first {num_frames_to_replay_in_current_segment} frames (Threshold: {current_threshold}).")
                print(f"  Replay target X from sequence (original stored max_x): {best_sequence_data['max_x']}, Cleared: {best_sequence_data['cleared']}")
            else:
                is_replaying_sequence = False
                print(f"Ep {episode_count}: Starting in EXPLORE mode (Replay segment too short or zero: {num_actual_replay_frames}. Full seq len: {len(full_actions_with_x)}, Threshold: {current_threshold}).")
        else:
            is_replaying_sequence = False
            print(f"Ep {episode_count}: Starting in EXPLORE mode (No successful sequences).")

        if not is_replaying_sequence: # 初期アクションを決定 (探索モードで開始する場合)
            # 最初の探索アクションは現在の許可セットから選ぶ
            current_allowed_action_list_init = ALLOWED_ACTIONS_SUBSETS_BY_X[current_action_set_config_idx]
            possible_actions_indices_in_initial_set_init = []
            for i, action_tuple_in_initial_init in enumerate(INITIAL_ACTION_SET_FOR_ENV):
                if action_tuple_in_initial_init in current_allowed_action_list_init:
                    possible_actions_indices_in_initial_set_init.append(i)
            
            if possible_actions_indices_in_initial_set_init:
                current_held_action_idx = random.choice(possible_actions_indices_in_initial_set_init)
            else: # 万が一候補がない場合
                current_held_action_idx = random.randrange(len(INITIAL_ACTION_SET_FOR_ENV))
            action_idx_for_this_frame = current_held_action_idx


        a_button_press_time_start = 0
        running_episode = True
        game_over_text_str = "GAME OVER"

        while running_episode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: env.close(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: env.close(); pygame.quit(); sys.exit()
                    if event.key == pygame.K_r:
                        running_episode = False; print(f"Ep {episode_count} reset by user.")
            if not running_episode: break

            current_x_pos_before_action = info.get('x_pos', 0)
            current_episode_max_x = max(current_episode_max_x, current_x_pos_before_action) # ★★★ フレームごとに更新

            # ★★★ X座標に基づくアクションセット設定の更新 ★★★
            next_config_idx_candidate = -1
            # current_episode_max_x を使うことで、一度進んだら後退してもアクションセットは戻らない
            for i in range(len(X_THRESHOLDS_FOR_ACTION_SET_SWITCH) -1, -1, -1):
                if current_episode_max_x >= X_THRESHOLDS_FOR_ACTION_SET_SWITCH[i]:
                    next_config_idx_candidate = i
                    break
            
            if next_config_idx_candidate != -1 and next_config_idx_candidate != current_action_set_config_idx:
                print(f"DEBUG: current_episode_max_x ({current_episode_max_x}) passed threshold {X_THRESHOLDS_FOR_ACTION_SET_SWITCH[next_config_idx_candidate]}. Switching action set config from {current_action_set_config_idx} to {next_config_idx_candidate}.")
                current_action_set_config_idx = next_config_idx_candidate
                CURRENT_ACTION_SET = ALLOWED_ACTIONS_SUBSETS_BY_X[current_action_set_config_idx]
                CURRENT_ACTION_SET_NAME = ACTION_SET_NAMES_BY_X[current_action_set_config_idx]
                update_pygame_caption()


            if is_replaying_sequence and apply_special_replay_exit_this_ep:
                if current_x_pos_before_action >= target_x_for_special_replay_exit:
                    print(f"DEBUG Replay Pre-Check: Current X ({current_x_pos_before_action}) >= Target X ({target_x_for_special_replay_exit}). Switching to EXPLORE.")
                    is_replaying_sequence = False
                    apply_special_replay_exit_this_ep = False
                    last_action_choice_time = time.time() - ACTION_INTERVAL - 0.01
                    action_idx_for_this_frame = -1

            if is_replaying_sequence:
                if replay_pointer < num_frames_to_replay_in_current_segment:
                    action_idx_for_this_frame, _ = current_replay_actions_with_x[replay_pointer]
                else:
                    is_replaying_sequence = False
                    print(f"  Replay segment finished ({replay_pointer}/{num_frames_to_replay_in_current_segment} frames). Switching to exploration for Ep {episode_count}.")
                    last_action_choice_time = time.time() - ACTION_INTERVAL - 0.01
                    action_idx_for_this_frame = -1

            if not is_replaying_sequence:
                time_since_last_choice = time.time() - last_action_choice_time
                if time_since_last_choice > ACTION_INTERVAL or action_idx_for_this_frame == -1:
                    last_action_choice_time = time.time()
                    
                    current_allowed_action_list = ALLOWED_ACTIONS_SUBSETS_BY_X[current_action_set_config_idx]
                    possible_actions_indices_in_initial_set = []
                    for i, action_tuple_in_initial in enumerate(INITIAL_ACTION_SET_FOR_ENV):
                        # action_tuple_in_initial は ['コマンド', 'コマンド', ...] の形
                        # current_allowed_action_list の要素も同じ形
                        if list(action_tuple_in_initial) in current_allowed_action_list : # list()で比較
                             possible_actions_indices_in_initial_set.append(i)
                        elif tuple(action_tuple_in_initial) in current_allowed_action_list: # tuple()でも比較（念のため）
                             possible_actions_indices_in_initial_set.append(i)


                    if not possible_actions_indices_in_initial_set:
                        print(f"WARNING: No allowed actions for current_x ({current_x_pos_before_action}), config_idx ({current_action_set_config_idx}). Defaulting to random from INITIAL_ACTION_SET_FOR_ENV.")
                        possible_actions_indices_in_initial_set = list(range(len(INITIAL_ACTION_SET_FOR_ENV)))
                    
                    effective_candidate_indices = possible_actions_indices_in_initial_set
                    # overall_best_x_pos ではなく current_episode_max_x を基準に「未知のエリア」かを判断することも検討可能
                    is_in_truly_unknown_area = current_x_pos_before_action > overall_best_x_pos
                    if not is_in_truly_unknown_area and short_term_failure_actions and SHORT_TERM_FAILURE_MEMORY_SIZE > 0:
                        filtered_by_failure = [i for i in effective_candidate_indices if i not in list(short_term_failure_actions)]
                        if filtered_by_failure:
                            effective_candidate_indices = filtered_by_failure
                    
                    if effective_candidate_indices:
                        current_held_action_idx = random.choice(effective_candidate_indices)
                    else: 
                        current_held_action_idx = random.choice(possible_actions_indices_in_initial_set) if possible_actions_indices_in_initial_set else random.randrange(len(INITIAL_ACTION_SET_FOR_ENV))
                    
                    action_idx_for_this_frame = current_held_action_idx
                else:
                    action_idx_for_this_frame = current_held_action_idx
            
            action_idx_to_step = action_idx_for_this_frame
            current_action_str_list_for_display = list(INITIAL_ACTION_SET_FOR_ENV[action_idx_to_step]) # 表示は常に INITIAL から

            if not is_replaying_sequence:
                temp_action_list_for_a_check = list(INITIAL_ACTION_SET_FOR_ENV[action_idx_to_step])
                if 'A' in temp_action_list_for_a_check:
                    if a_button_press_time_start == 0:
                        a_button_press_time_start = time.time()
                    elif time.time() - a_button_press_time_start > A_BUTTON_MAX_HOLD_TIME:
                        temp_action_list_for_a_check_released_A = [token for token in temp_action_list_for_a_check if token != 'A']
                        a_button_press_time_start = 0
                        try:
                            action_idx_to_step = INITIAL_ACTION_SET_FOR_ENV.index(temp_action_list_for_a_check_released_A)
                        except ValueError:
                            if not temp_action_list_for_a_check_released_A:
                                noop_action_tuple = ['NOOP']
                                if noop_action_tuple in INITIAL_ACTION_SET_FOR_ENV: # NOOPがリスト形式の場合
                                    try: action_idx_to_step = INITIAL_ACTION_SET_FOR_ENV.index(noop_action_tuple)
                                    except ValueError: pass
                                elif tuple(noop_action_tuple) in INITIAL_ACTION_SET_FOR_ENV: # NOOPがタプル形式の場合
                                    try: action_idx_to_step = INITIAL_ACTION_SET_FOR_ENV.index(tuple(noop_action_tuple))
                                    except ValueError: pass
                        current_action_str_list_for_display = list(INITIAL_ACTION_SET_FOR_ENV[action_idx_to_step])
                elif a_button_press_time_start != 0:
                    a_button_press_time_start = 0
            
            pressed_buttons_for_draw = set()
            if current_action_str_list_for_display and current_action_str_list_for_display != ['NOOP']:
                for token in current_action_str_list_for_display:
                    if token in COMMAND_TO_KEY:
                        pressed_buttons_for_draw.add(COMMAND_TO_KEY[token])

            if not done:
                next_state_frame, reward, terminated, truncated, current_step_info = env.step(action_idx_to_step)
                current_x_pos_after_action = current_step_info.get('x_pos', 0)
                episode_frame_by_frame_log.append((action_idx_to_step, current_x_pos_after_action))

                if previous_x_pos - current_x_pos_after_action >= 300:
                    print(f"DEBUG: Loop Detected! Ep: {episode_count}, PrevX(before step): {previous_x_pos}, CurrX(after step): {current_x_pos_after_action}")
                    new_loop_warp_x = current_x_pos_after_action
                    overall_best_x_pos = new_loop_warp_x
                    g_x_pos_at_loop_warp = new_loop_warp_x
                    g_special_replay_control_active = True
                    print(f"DEBUG: Loop Set - overall_best_x_pos: {overall_best_x_pos}, g_x_pos_at_loop_warp: {g_x_pos_at_loop_warp}, g_special_replay_control_active: {g_special_replay_control_active}, x_pos_at_loop_start_approx: {previous_x_pos}")

                    if successful_sequences:
                        current_best_seq_data = successful_sequences[0]
                        original_sequence_with_x = current_best_seq_data.get("sequence_with_x", [])
                        truncated_sequence_with_x = []
                        new_max_x_for_truncated_seq = 0
                        sequence_was_truncated = False
                        found_x_data_in_seq = any(x_val != -1 for _, x_val in original_sequence_with_x) if original_sequence_with_x else False

                        for i, (action_idx_seq, x_val) in enumerate(original_sequence_with_x):
                            if found_x_data_in_seq and x_val != -1 and x_val >= g_x_pos_at_loop_warp :
                                print(f"DEBUG: Truncating successful_sequence at index {i} (X={x_val} >= g_x_pos_at_loop_warp={g_x_pos_at_loop_warp}).")
                                sequence_was_truncated = True
                                break
                            truncated_sequence_with_x.append((action_idx_seq, x_val))
                            if x_val != -1: new_max_x_for_truncated_seq = max(new_max_x_for_truncated_seq, x_val)
                            elif i > 0 and original_sequence_with_x : # X不明でも直前が有効なら
                                prev_x_val_in_seq = original_sequence_with_x[i-1][1]
                                if prev_x_val_in_seq != -1: new_max_x_for_truncated_seq = max(new_max_x_for_truncated_seq, prev_x_val_in_seq)
                        
                        if sequence_was_truncated :
                            if not truncated_sequence_with_x:
                                print(f"DEBUG: Successful sequence entirely truncated. Clearing successful_sequences.")
                                successful_sequences.clear()
                            else:
                                current_best_seq_data["sequence_with_x"] = truncated_sequence_with_x
                                actual_max_x_after_trunc = new_max_x_for_truncated_seq
                                current_best_seq_data["max_x"] = min(actual_max_x_after_trunc, g_x_pos_at_loop_warp -1 if g_x_pos_at_loop_warp > 0 else 0)
                                print(f"DEBUG: Successful sequence truncated by X. New len: {len(current_best_seq_data['sequence_with_x'])}, New max_x: {current_best_seq_data['max_x']}")
                        elif not found_x_data_in_seq and current_best_seq_data.get("max_x",0) > g_x_pos_at_loop_warp:
                            print(f"DEBUG: Loop Detected. No X data in seq or seq not truncated by X. Adjusting stored max_x from {current_best_seq_data.get('max_x',0)} to {g_x_pos_at_loop_warp -1}")
                            current_best_seq_data["max_x"] = g_x_pos_at_loop_warp -1 if g_x_pos_at_loop_warp > 0 else 0
                        else:
                             print(f"DEBUG: Loop Detected. Sequence not truncated by X. Original max_x: {current_best_seq_data.get('max_x',0)}")


                    if SHORT_TERM_FAILURE_MEMORY_SIZE > 0 : short_term_failure_actions.clear()
                    info = current_step_info
                    info['loop_detected_event'] = True
                    done = True
                else:
                    info = current_step_info
                    done = terminated or truncated
                    total_episode_reward += reward
                    episode_step_count += 1
                    state_frame = next_state_frame
                    # current_episode_max_x はフレーム開始時に更新済み
                    previous_x_pos = current_x_pos_after_action

                if is_replaying_sequence and not done:
                    replay_pointer += 1

            screen.fill(BACKGROUND_COLOR)
            if state_frame is not None:
                game_surface_original_res = convert_frame_to_pygame_surface(state_frame)
                game_surface_scaled = pygame.transform.scale(game_surface_original_res, GAME_SCREEN_RECT.size)
                screen.blit(game_surface_scaled, GAME_SCREEN_RECT.topleft)
            if controller_base_image_scaled:
                draw_controller_state_ui(screen, controller_base_image_scaled, pressed_buttons_for_draw, CONTROLLER_RECT)

            draw_text_info(screen, current_action_str_list_for_display, total_episode_reward, episode_step_count, episode_count, info.get('x_pos',0), is_replaying_sequence, current_threshold)

            if done:
                if info.get('flag_get', False): game_over_text_str = "STAGE CLEAR!"
                elif info.get('loop_detected_event', False): game_over_text_str = "LOOP DETECTED"
                elif info.get('time', 400) <= 1: game_over_text_str = "TIME UP"
                else: game_over_text_str = "GAME OVER"
                update_memory_at_episode_end(info, episode_frame_by_frame_log, current_episode_max_x, total_episode_reward)
                game_over_surf = font_medium.render(game_over_text_str, True, (255, 60, 60))
                text_rect = game_over_surf.get_rect(center=GAME_SCREEN_RECT.center)
                screen.blit(game_over_surf, text_rect)
                next_ep_surf = font_small.render("Press R to restart, ESC to Quit. Waiting for next...", True, (200,200,0))
                next_ep_rect = next_ep_surf.get_rect(centerx=GAME_SCREEN_RECT.centerx, top=text_rect.bottom + 10)
                screen.blit(next_ep_surf, next_ep_rect)
                pygame.display.flip()
                wait_start_time = time.time()
                r_pressed_during_wait = False
                while time.time() - wait_start_time < 1.0:
                    for event_wait in pygame.event.get():
                        if event_wait.type == pygame.QUIT: env.close(); pygame.quit(); sys.exit()
                        if event_wait.type == pygame.KEYDOWN:
                            if event_wait.key == pygame.K_ESCAPE: env.close(); pygame.quit(); sys.exit()
                            if event_wait.key == pygame.K_r: r_pressed_during_wait = True; break
                    if r_pressed_during_wait: break
                    clock.tick(FPS)
                running_episode = False
            pygame.display.flip()
            clock.tick(FPS)

        if not running_episode and not done:
             print(f"Episode {episode_count} was manually reset.")
        elif done:
             print(f"Episode {episode_count} finished. Outcome: {game_over_text_str}, Reward: {total_episode_reward:.0f}, Max_X_this_ep: {current_episode_max_x}, Steps: {episode_step_count}, Flag: {info.get('flag_get', False)}")
             print(f"  End of ep {episode_count}: overall_best_x_pos: {overall_best_x_pos}, g_x_pos_at_loop_warp: {g_x_pos_at_loop_warp}, g_special_replay_control_active: {g_special_replay_control_active}")
             if successful_sequences:
                 print(f"  End of ep: successful_sequences[0]['max_x']: {successful_sequences[0]['max_x']}, len: {len(successful_sequences[0]['sequence_with_x'])}")

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    init_pygame()
    mario_environment = init_mario_env()
    game_loop(mario_environment)