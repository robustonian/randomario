#!/usr/bin/env python3
"""
Mario Stage Progress Viewer GUI
Displays completion status for stages 1-1 through 8-4 in a 4x8 matrix.
"""

import os
import pickle
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import tkinter as tk
from tkinter import ttk

# Add current directory to path for importing test.py
sys.path.insert(0, '.')

# Try to import Cell from test.py, fallback to mock if it fails
try:
    from test import Cell
except ImportError:
    # Mock Cell class to handle pickle loading from test.py
    @dataclass
    class Cell:
        cell_id: Tuple[int, int, str]  # (x_bin, y_bin, status)
        path: list = field(default_factory=list)  # Action sequence from start to reach this cell
        max_x: int = 0  # Maximum x reached when reaching this cell
        visits: int = 0  # Number of times used as starting cell
        created_at: float = field(default_factory=time.time)


class ProgressData:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.exists = os.path.exists(file_path)
        self.episodes_done = 0
        self.first_clear_episode: Optional[int] = None
        self.last_modified = 0
        
        if self.exists:
            try:
                self.last_modified = os.path.getmtime(file_path)
                # Try pickle loading with error recovery
                with open(file_path, 'rb') as f:
                    import pickle
                    # Use pickle.HIGHEST_PROTOCOL to handle any protocol version
                    data = pickle.load(f)
                self.episodes_done = data.get('episodes_done', 0)
                self.first_clear_episode = data.get('first_clear_episode', None)
            except Exception as e:
                # If pickle fails, assume file exists but extract basic info
                print(f"Pickle loading failed for {os.path.basename(file_path)}, using fallback")
                self.episodes_done = 1  # At least attempted since file exists
                self.first_clear_episode = None
                self.exists = True  # Keep as existing since file is there

    def is_modified(self) -> bool:
        if not os.path.exists(self.file_path):
            return self.exists  # File was deleted
        
        current_mtime = os.path.getmtime(self.file_path)
        return current_mtime != self.last_modified

    def update(self):
        if not os.path.exists(self.file_path):
            if self.exists:
                # File was deleted
                self.exists = False
                self.episodes_done = 0
                self.first_clear_episode = None
                self.last_modified = 0
            return
        
        try:
            self.last_modified = os.path.getmtime(self.file_path)
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            self.episodes_done = data.get('episodes_done', 0)
            self.first_clear_episode = data.get('first_clear_episode', None)
            self.exists = True
        except Exception:
            # Fall back to basic info if pickle loading fails
            self.episodes_done = 1  # At least attempted
            self.first_clear_episode = None
            self.exists = True

    @property
    def status_text(self) -> str:
        if not self.exists:
            return "-"
        elif self.first_clear_episode is not None:
            return f"Cleared\nEP{self.first_clear_episode}"
        else:
            return f"Playing\nEP{self.episodes_done}"

    @property
    def status_icon(self) -> str:
        if not self.exists:
            return "⚫"  # Not attempted
        elif self.first_clear_episode is not None:
            return "✅"  # Cleared
        else:
            return "🔄"  # In progress

    @property
    def bg_color(self) -> str:
        if not self.exists:
            return "#34495e"  # Dark gray
        elif self.first_clear_episode is not None:
            return "#27ae60"  # Green
        else:
            return "#f39c12"  # Orange
    
    @property
    def text_color(self) -> str:
        if not self.exists:
            return "#7f8c8d"  # Muted gray
        elif self.first_clear_episode is not None:
            return "#ecf0f1"  # White
        else:
            return "#2c3e50"  # Dark text


class ProgressViewerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🍄 Mario Progress Dashboard")
        self.root.geometry("900x400")
        self.root.configure(bg='#2c3e50')
        self.root.resizable(True, False)
        
        # Data storage
        self.progress_data: Dict[str, ProgressData] = {}
        self.labels: Dict[str, ttk.Label] = {}
        
        # Create GUI
        self.setup_gui()
        
        # Initialize data
        self.refresh_all_data()
        
        # Start file monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_files, daemon=True)
        self.monitor_thread.start()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        
        # Title
        title_label = tk.Label(self.root, text="🍄 Super Mario Bros. Progress Dashboard",
                             font=('Segoe UI', 14, 'bold'),
                             bg='#2c3e50', fg='#ecf0f1')
        title_label.pack(pady=(10, 5))
        
        # Info text with better styling
        info_text = "✅ Cleared   🔄 Playing   ⚫ Not Started"
        info_label = tk.Label(self.root, text=info_text,
                            font=('Segoe UI', 9),
                            bg='#2c3e50', fg='#bdc3c7')
        info_label.pack(pady=(0, 10))
        
        # Main frame for the grid
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=15, pady=(0, 10))
        
        # Create 8x4 grid (8 worlds, 4 stages each) - horizontal layout
        for world in range(1, 9):  # Worlds 1-8
            world_frame = tk.LabelFrame(main_frame, text=f"World {world}", 
                                      bg='#34495e', fg='#ecf0f1',
                                      font=('Segoe UI', 9, 'bold'),
                                      bd=1, relief='solid')
            world_frame.grid(row=0, column=world-1, padx=2, pady=5, sticky='nsew')
            main_frame.columnconfigure(world-1, weight=1)
            
            for stage in range(1, 5):  # Stages 1-4
                stage_key = f"{world}-{stage}"
                
                # Create compact frame for each stage
                stage_frame = tk.Frame(world_frame, bg='#34495e', relief='flat', bd=1)
                stage_frame.grid(row=stage-1, column=0, padx=1, pady=1, sticky='ew')
                world_frame.rowconfigure(stage-1, weight=1)
                world_frame.columnconfigure(0, weight=1)
                
                # Stage label with compact layout
                stage_label = tk.Label(stage_frame, text=f"{world}-{stage}", 
                                     font=('Segoe UI', 8, 'bold'),
                                     bg='#34495e', fg='#ecf0f1', pady=1)
                stage_label.pack()
                
                # Status label (icon + text) with better styling
                status_label = tk.Label(stage_frame, text="⚫\n-", 
                                      font=('Segoe UI', 7),
                                      bg='#34495e', fg='#bdc3c7', 
                                      pady=2, justify='center')
                status_label.pack()
                
                self.labels[stage_key] = status_label
        
        main_frame.rowconfigure(0, weight=1)
        
        # Bottom panel with button and status
        bottom_frame = tk.Frame(self.root, bg='#2c3e50')
        bottom_frame.pack(fill='x', padx=15, pady=(5, 10))
        
        # Styled refresh button
        refresh_btn = tk.Button(bottom_frame, text="🔄 Refresh",
                              font=('Segoe UI', 9), bg='#3498db', fg='white',
                              relief='flat', bd=0, padx=10, pady=3,
                              activebackground='#2980b9', activeforeground='white',
                              command=self.refresh_all_data)
        refresh_btn.pack(side='left')
        
        # Status bar with better styling
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Monitoring files...")
        status_bar = tk.Label(bottom_frame, textvariable=self.status_var,
                            font=('Segoe UI', 8), bg='#2c3e50', fg='#95a5a6',
                            anchor='w')
        status_bar.pack(side='right', fill='x', expand=True, padx=(10, 0))

    def get_archive_path(self, stage: str, action_set: str = "right_only") -> str:
        return f"pkl/go_explore_archive_{stage}_{action_set}.pkl"

    def refresh_all_data(self):
        updated_count = 0
        
        for world in range(1, 9):
            for stage in range(1, 5):
                stage_key = f"{world}-{stage}"
                
                # Try different action sets (right_only first, then complex)
                archive_path = None
                for action_set in ["right_only", "complex", "simple"]:
                    path = self.get_archive_path(stage_key, action_set)
                    if os.path.exists(path):
                        archive_path = path
                        break
                
                if archive_path is None:
                    archive_path = self.get_archive_path(stage_key, "right_only")
                
                # Update or create progress data
                if stage_key not in self.progress_data:
                    self.progress_data[stage_key] = ProgressData(archive_path)
                    updated_count += 1
                elif self.progress_data[stage_key].file_path != archive_path:
                    self.progress_data[stage_key] = ProgressData(archive_path)
                    updated_count += 1
                elif self.progress_data[stage_key].is_modified():
                    self.progress_data[stage_key].update()
                    updated_count += 1
        
        # Update GUI
        self.update_display()
        
        if updated_count > 0:
            self.status_var.set(f"Updated {updated_count} stages at {time.strftime('%H:%M:%S')}")

    def update_display(self):
        for stage_key, progress in self.progress_data.items():
            if stage_key in self.labels:
                label = self.labels[stage_key]
                text = f"{progress.status_icon}\n{progress.status_text}"
                
                # Update label with new colors and text
                label.config(text=text, 
                           bg=progress.bg_color, 
                           fg=progress.text_color)
                
                # Update parent frame background too
                try:
                    parent_frame = label.master
                    parent_frame.config(bg=progress.bg_color)
                except:
                    pass

    def monitor_files(self):
        """Background thread to monitor file changes"""
        while self.monitoring:
            try:
                # Check for changes every 2 seconds
                time.sleep(2)
                
                if not self.monitoring:
                    break
                
                # Check if any files have changed
                changes_detected = False
                for progress in self.progress_data.values():
                    if progress.is_modified():
                        changes_detected = True
                        break
                
                if changes_detected:
                    # Schedule GUI update on main thread
                    self.root.after(0, self.refresh_all_data)
                    
            except Exception as e:
                print(f"Error in file monitoring: {e}")
                time.sleep(5)  # Wait longer if there's an error

    def on_closing(self):
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    # Ensure pkl directory exists
    pkl_dir = Path("pkl")
    if not pkl_dir.exists():
        print("Warning: pkl/ directory does not exist. Creating it...")
        pkl_dir.mkdir()
    
    # Start GUI
    app = ProgressViewerGUI()
    app.run()


if __name__ == "__main__":
    main()