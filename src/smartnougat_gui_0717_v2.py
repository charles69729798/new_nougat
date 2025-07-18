#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartNougat GUI 0717 v2 - Improved version with no redundancy
- Uses smartnougat_0715.py for processing
- Converts result_viewer_0715.html to standalone HTML
- Direct output to C:\\Nougat_result\\filename_date_time\\
- Progress shows processed/total formulas
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import re
import time
import queue
from pathlib import Path
import json
import base64
from datetime import datetime
from bs4 import BeautifulSoup
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

class SmartNougatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartNougat GUI 0717 v2 - PDF/DOCX to LaTeX Converter")
        self.root.geometry("900x750")
        
        # Load configuration
        self.config = self.load_config()
        
        # Variables
        self.selected_file = tk.StringVar()
        self.page_range = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.final_html_path = tk.StringVar()
        self.is_processing = False
        self.process = None
        self.output_queue = queue.Queue()
        self.start_time = None
        
        # Statistics
        self.current_page = tk.IntVar(value=0)
        self.total_pages = tk.IntVar(value=0)
        self.processed_formulas = tk.IntVar(value=0)
        self.total_formulas = tk.IntVar(value=0)
        self.processing_time = tk.StringVar(value="0초")
        self.progress_percent = tk.IntVar(value=0)
        
        # Page-specific formula tracking
        self.current_page_formulas = tk.IntVar(value=0)
        self.current_page_processed = tk.IntVar(value=0)
        self.page_progress_percent = tk.IntVar(value=0)
        
        # Create UI
        self.create_widgets()
        
        # Start output reader thread
        self.root.after(100, self.check_output_queue)
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # File selection section
        ttk.Label(main_frame, text="Select File:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=0, padx=(0, 5))
        
        # File entry with drag and drop support
        self.file_entry = ttk.Entry(file_frame, textvariable=self.selected_file, state='readonly')
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Enable drag and drop if available
        if HAS_DND:
            self.enable_drag_drop()
        
        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(1, weight=1)
        
        ttk.Label(options_frame, text="Page Range:").grid(row=0, column=0, sticky=tk.W, pady=5)
        page_entry = ttk.Entry(options_frame, textvariable=self.page_range, width=20)
        page_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(options_frame, text="(e.g., 1-5, 10, 15 or leave empty for all)", 
                 font=('Arial', 8), foreground='gray').grid(row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(options_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_entry = ttk.Entry(options_frame, textvariable=self.output_folder, state='readonly')
        self.output_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="Real-time Information", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(3, weight=1)
        
        # Overall progress bar
        progress_container = ttk.Frame(stats_frame)
        progress_container.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        progress_container.columnconfigure(1, weight=1)
        
        ttk.Label(progress_container, text="Progress:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_container, maximum=100, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        self.progress_label = ttk.Label(progress_container, text="0%")
        self.progress_label.grid(row=0, column=2, padx=(10, 0))
        
        # Page progress bar (formulas within current page)
        page_progress_container = ttk.Frame(stats_frame)
        page_progress_container.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        page_progress_container.columnconfigure(1, weight=1)
        
        ttk.Label(page_progress_container, text="Page Progress:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.page_progress_bar = ttk.Progressbar(page_progress_container, maximum=100, mode='determinate')
        self.page_progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        self.page_progress_label = ttk.Label(page_progress_container, text="0/0")
        self.page_progress_label.grid(row=0, column=2, padx=(10, 0))
        
        # Statistics labels
        ttk.Label(stats_frame, text="Pages:").grid(row=2, column=0, sticky=tk.W)
        self.page_label = ttk.Label(stats_frame, text="0 / 0")
        self.page_label.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Processed:").grid(row=2, column=2, sticky=tk.E)
        self.formula_label = ttk.Label(stats_frame, text="0 formulas")
        self.formula_label.grid(row=2, column=3, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Time:").grid(row=3, column=0, sticky=tk.W)
        self.time_label = ttk.Label(stats_frame, textvariable=self.processing_time)
        self.time_label.grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Status:").grid(row=3, column=2, sticky=tk.E)
        self.status_label = ttk.Label(stats_frame, text="Ready")
        self.status_label.grid(row=3, column=3, sticky=tk.W)
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for coloring
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('info', foreground='blue')
        self.log_text.tag_config('warning', foreground='orange')
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_processing, 
                                      style='Accent.TButton')
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, 
                                     state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.view_html_button = ttk.Button(button_frame, text="View HTML", command=self.view_html, 
                                          state=tk.DISABLED)
        self.view_html_button.grid(row=0, column=2, padx=5)
        
        self.open_folder_button = ttk.Button(button_frame, text="Open Folder", command=self.open_output_folder, 
                                            state=tk.DISABLED)
        self.open_folder_button.grid(row=0, column=3, padx=5)
        
        # Debug button for testing progress - always show
        ttk.Button(button_frame, text="Test Progress", command=self.test_progress).grid(row=0, column=4, padx=5)
        
        # Create style for accent button
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    
    def load_config(self):
        """Load configuration from config.json"""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Default configuration
        return {
            "mathjax": {
                "local_path": "C:\\SmartNougat\\resources\\mathjax\\es5\\tex-svg.js",
                "relative_path": "../resources/mathjax/es5/tex-svg.js",
                "cdn_fallback": "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
            }
        }
    
    def enable_drag_drop(self):
        """Enable drag and drop functionality"""
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.drop_file)
            
            # Visual feedback for drag and drop area
            drop_label = ttk.Label(self.root, text="Drop files here", 
                                  font=('Arial', 9), foreground='gray')
            drop_label.place(relx=0.5, rely=0.1, anchor='center')
        except:
            pass
    
    def drop_file(self, event):
        """Handle dropped file"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            if file_path.lower().endswith(('.pdf', '.docx')):
                self.selected_file.set(file_path)
                self.update_output_folder()
                self.log(f"File dropped: {file_path}", 'info')
            else:
                messagebox.showwarning("Warning", "Only PDF or DOCX files are supported.")
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select PDF or DOCX file",
            filetypes=[
                ("Document Files", "*.pdf;*.docx"),
                ("PDF Files", "*.pdf"),
                ("DOCX Files", "*.docx"),
                ("All Files", "*.*")
            ]
        )
        
        if filename:
            self.selected_file.set(filename)
            self.update_output_folder()
            self.log(f"File selected: {filename}", 'info')
    
    def update_output_folder(self):
        """Update output folder based on selected file"""
        if self.selected_file.get():
            file_stem = Path(self.selected_file.get()).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"C:\\Nougat_result\\{file_stem}_{timestamp}"
            self.output_folder.set(output_path)
    
    def log(self, message, tag=None):
        """Add message to log with optional formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message, tag)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_processing(self):
        if not self.selected_file.get():
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.view_html_button.config(state=tk.DISABLED)
        self.open_folder_button.config(state=tk.DISABLED)
        
        # Reset statistics
        self.current_page.set(0)
        self.total_pages.set(0)
        self.processed_formulas.set(0)
        self.total_formulas.set(0)
        self.progress_percent.set(0)
        self.current_page_formulas.set(0)
        self.current_page_processed.set(0)
        self.page_progress_percent.set(0)
        self.page_progress_bar['value'] = 0
        self.page_progress_label.config(text="0/0")
        self.start_time = time.time()
        self.final_html_path.set("")
        self.status_label.config(text="Processing...")
        
        # Reset duplicate tracking
        self._last_formula_line = None
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("Starting processing...", 'info')
        
        # Start processing thread
        thread = threading.Thread(target=self.run_smartnougat, daemon=True)
        thread.start()
        
        # Start timer update
        self.update_timer()
    
    def run_smartnougat(self):
        try:
            # Build command - use smartnougat_0715.py with direct output path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            smartnougat_path = os.path.join(script_dir, "smartnougat_0715.py")
            cmd = [sys.executable, smartnougat_path, self.selected_file.get()]
            
            # Add page range if specified
            if self.page_range.get():
                cmd.extend(["-p", self.page_range.get()])
            
            # Output to parent directory - smartnougat will create its own subfolder
            parent_output = str(Path(self.output_folder.get()).parent)
            cmd.extend(["-o", parent_output])
            
            self.output_queue.put(("info", f"Running command: {' '.join(cmd)}"))
            
            # Start process - capture both stdout and stderr
            # Use CREATE_NO_WINDOW on Windows to prevent console window
            creationflags = 0
            if sys.platform == 'win32':
                try:
                    creationflags = subprocess.CREATE_NO_WINDOW
                except AttributeError:
                    creationflags = 0x08000000  # CREATE_NO_WINDOW constant
                
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationflags
            )
            
            # Read output line by line
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    # Put line in queue for parsing in main thread
                    self.output_queue.put(("parse", line))
            
            self.process.wait()
            
            if self.process.returncode == 0:
                self.output_queue.put(("success", "Processing complete! Generating standalone HTML..."))
                self.status_label.config(text="Converting to HTML...")
                # Convert to standalone HTML
                self.create_standalone_html()
            else:
                self.output_queue.put(("error", f"Processing failed (code: {self.process.returncode})"))
                self.status_label.config(text="Processing failed")
                
        except Exception as e:
            self.output_queue.put(("error", f"Error occurred: {str(e)}"))
            self.status_label.config(text="Error occurred")
        finally:
            self.is_processing = False
            self.root.after(0, self.processing_finished)
    
    def create_standalone_html(self):
        """Convert result_viewer_0715.html to standalone HTML with embedded images"""
        try:
            # Find the actual output directory created by smartnougat_0715.py
            # smartnougat_0715.py saves to src/output/ directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            actual_output_parent = os.path.join(script_dir, "output")
            
            actual_output_dir = None
            base_name = Path(self.selected_file.get()).stem
            
            # Debug: Show what we're looking for
            self.output_queue.put(("info", f"Looking for output in: {actual_output_parent}"))
            self.output_queue.put(("info", f"Base name: {base_name}"))
            
            # Look for directory with pattern: base_name_smartnougat_timestamp
            # Get all folders and sort by modification time to get the most recent
            found_folders = []
            all_folders_with_time = []
            
            for folder in os.listdir(actual_output_parent):
                folder_path = os.path.join(actual_output_parent, folder)
                if os.path.isdir(folder_path) and folder.startswith(base_name):
                    mod_time = os.path.getmtime(folder_path)
                    all_folders_with_time.append((folder, mod_time))
                    found_folders.append(folder)
            
            # Sort by modification time (newest first)
            all_folders_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Use the most recent folder
            if all_folders_with_time:
                most_recent_folder = all_folders_with_time[0][0]
                actual_output_dir = os.path.join(actual_output_parent, most_recent_folder)
                self.output_queue.put(("info", f"Using most recent folder: {most_recent_folder}"))
            
            self.output_queue.put(("info", f"Found folders: {found_folders}"))
            
            if not actual_output_dir:
                self.output_queue.put(("error", "Output directory not found."))
                return
            
            # Read result_viewer_0715.html
            viewer_html_path = os.path.join(actual_output_dir, "result_viewer_0715.html")
            if not os.path.exists(viewer_html_path):
                self.output_queue.put(("error", "result_viewer_0715.html not found."))
                return
            
            with open(viewer_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Convert local image paths to base64
            images_dir = os.path.join(actual_output_dir, "images")
            image_count = 0
            for img_tag in soup.find_all('img'):
                src = img_tag.get('src')
                if src and not src.startswith('data:'):
                    # Extract filename from src
                    img_filename = os.path.basename(src)
                    img_path = os.path.join(images_dir, img_filename)
                    
                    if os.path.exists(img_path):
                        with open(img_path, 'rb') as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                            img_tag['src'] = f"data:image/png;base64,{img_base64}"
                            image_count += 1
            
            self.output_queue.put(("info", f"Embedded {image_count} images"))
            
            # Add zoom functionality BEFORE MathJax embedding
            self.add_zoom_functionality(soup)
            self.output_queue.put(("info", "Added zoom functionality"))
            
            # Embed MathJax directly into HTML
            mathjax_embedded = False
            for script in soup.find_all('script'):
                src = script.get('src')
                if src and 'mathjax' in src.lower():
                    mathjax_content = None
                    
                    # First try local MathJax file
                    local_mathjax_paths = [
                        r"C:\SmartNougat\resources\mathjax\es5\tex-svg.js",
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "mathjax", "es5", "tex-svg.js")
                    ]
                    
                    for local_mathjax_path in local_mathjax_paths:
                        if os.path.exists(local_mathjax_path):
                            try:
                                self.output_queue.put(("info", f"Loading local MathJax from: {local_mathjax_path}"))
                                with open(local_mathjax_path, 'r', encoding='utf-8') as f:
                                    mathjax_content = f.read()
                                break
                            except Exception as e:
                                self.output_queue.put(("warning", f"Failed to read local MathJax: {str(e)}"))
                    
                    # If no local file, download from CDN
                    if not mathjax_content:
                        try:
                            import urllib.request
                            self.output_queue.put(("info", "Downloading MathJax from CDN..."))
                            
                            cdn_url = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
                            with urllib.request.urlopen(cdn_url) as response:
                                mathjax_content = response.read().decode('utf-8')
                        except Exception as e:
                            self.output_queue.put(("warning", f"Failed to download MathJax: {str(e)}"))
                    
                    # Embed MathJax if we have content
                    if mathjax_content:
                        # Create a new script tag with embedded content
                        new_script = soup.new_tag('script')
                        new_script.string = mathjax_content
                        
                        # Copy other attributes (like id) but not src or async
                        for attr, value in script.attrs.items():
                            if attr not in ['src', 'async']:
                                new_script[attr] = value
                        
                        # Replace the old script tag with the new one
                        script.replace_with(new_script)
                        mathjax_embedded = True
                        self.output_queue.put(("success", f"MathJax embedded successfully ({len(mathjax_content):,} bytes)"))
                    else:
                        self.output_queue.put(("info", "HTML will require internet connection for MathJax"))
                        # Keep the CDN link
                        script['src'] = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
            
            if not mathjax_embedded:
                # MathJax is required - add it manually if not found
                self.output_queue.put(("warning", "MathJax script tag not found, adding it manually"))
                
                # Add MathJax configuration
                config_script = soup.new_tag('script')
                config_script.string = """
                window.MathJax = {
                    tex: {
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                        processEscapes: true
                    },
                    svg: {
                        fontCache: 'global'
                    }
                };
                """
                
                # Find head or create one
                head = soup.find('head')
                if not head:
                    head = soup.new_tag('head')
                    soup.insert(0, head)
                
                head.append(config_script)
                
                # Download and embed MathJax
                try:
                    import urllib.request
                    self.output_queue.put(("info", "Downloading MathJax (required)..."))
                    
                    cdn_url = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
                    with urllib.request.urlopen(cdn_url) as response:
                        mathjax_content = response.read().decode('utf-8')
                    
                    mathjax_script = soup.new_tag('script')
                    mathjax_script.string = mathjax_content
                    head.append(mathjax_script)
                    
                    self.output_queue.put(("success", f"MathJax embedded successfully ({len(mathjax_content):,} bytes)"))
                except Exception as e:
                    self.output_queue.put(("error", f"Failed to embed MathJax: {str(e)}"))
                    self.output_queue.put(("error", "Output HTML will not render math properly!"))
            
            # Save standalone HTML
            file_stem = Path(self.selected_file.get()).stem
            date_str = datetime.now().strftime("%Y%m%d")
            html_filename = f"{file_stem}_{date_str}.html"
            
            # Create final output directory if it doesn't exist
            os.makedirs(self.output_folder.get(), exist_ok=True)
            html_path = os.path.join(self.output_folder.get(), html_filename)
            
            # Write the final HTML
            final_html = str(soup)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(final_html)
            
            # Report file size
            file_size = os.path.getsize(html_path)
            self.output_queue.put(("info", f"Final HTML size: {file_size:,} bytes"))
            
            self.final_html_path.set(html_path)
            self.output_queue.put(("success", f"Standalone HTML created: {html_path}"))
            self.status_label.config(text="Complete")
            
        except Exception as e:
            import traceback
            self.output_queue.put(("error", f"HTML conversion error: {str(e)}"))
            self.output_queue.put(("error", f"Traceback: {traceback.format_exc()}"))
            self.status_label.config(text="Conversion error")
    
    def parse_output(self, line):
        """Parse output line and extract information"""
        self.output_queue.put(("normal", line))
        
        # Extract page information - simplified pattern
        page_match = re.search(r'Processing page\s+(\d+)/(\d+)', line)
        if page_match:
            current, total = map(int, page_match.groups())
            
            # Calculate progress based on page START
            # Page 1 starts = 0%, Page 2 starts = 25%, etc.
            # But last page completion = 100%
            if current < total:
                percent = int(((current - 1) / total) * 100)
            else:
                # For last page, we'll update to 100% when processing completes
                percent = int(((current - 1) / total) * 100)
            
            self.current_page.set(current)
            self.total_pages.set(total)
            self.update_page_label()
            
            # Reset page-specific formula counters when starting new page
            self.current_page_formulas.set(0)
            self.current_page_processed.set(0)
            self.page_progress_bar['value'] = 0
            self.page_progress_label.config(text="0/0")
            
            # Queue progress update to be handled in main thread
            self.output_queue.put(("progress", percent))
            self.output_queue.put(("info", f"Starting page {current}/{total} - Progress: {percent}%"))
            return  # Important: return here to avoid duplicate processing
        
        # Extract total formula count - match the exact log pattern
        formula_match = re.search(r'Detected\s+(\d+)\s+formulas\s+on\s+page', line)
        if not formula_match:
            formula_match = re.search(r'Found\s+(\d+)\s+formulas?', line, re.IGNORECASE)
        if formula_match:
            count = int(formula_match.group(1))
            # Skip duplicate log lines by checking if this line was already processed
            if not hasattr(self, '_last_formula_line') or self._last_formula_line != line:
                self._last_formula_line = line
                # Add to total formulas (accumulate across pages)
                current_total = self.total_formulas.get()
                new_total = current_total + count
                self.total_formulas.set(new_total)
                self.update_formula_label()
                
                # Set current page formula count
                self.current_page_formulas.set(count)
                self.current_page_processed.set(0)
                self.update_page_progress()
                
                # Debug info
                self.output_queue.put(("info", f"Detected {count} formulas on this page, total so far: {new_total}"))
        
        # Track processed formulas - look for the specific log pattern
        formula_process_match = re.search(r'Processing formula (\d+)/(\d+):', line)
        if formula_process_match:
            current_formula, page_formulas = map(int, formula_process_match.groups())
            # Update total processed count
            self.processed_formulas.set(self.processed_formulas.get() + 1)
            self.update_formula_label()
            
            # Update page-specific progress
            self.current_page_processed.set(current_formula)
            self.update_page_progress()
            
            # Debug output
            self.output_queue.put(("info", f"Page formula {current_formula}/{page_formulas} - Total: {self.processed_formulas.get()}"))
            return
            
        # Also track warning messages that indicate formula processing
        if 'The following generation flags' in line or 'Passing a tuple of' in line:
            # Don't increment here since we'll get the exact count from the log above
            self.output_queue.put(("info", "Nougat processing..."))
        
        # Check for completion of last page
        if 'Cache and memory cleanup complete' in line and self.total_pages.get() > 0:
            # Last page completed - set to 100%
            self.output_queue.put(("progress", 100))
            self.output_queue.put(("success", "All pages processed - 100%"))
            return
            
        # Check for errors
        if 'error' in line.lower():
            self.output_queue.put(("error", line))
    
    def update_progress(self):
        """Update progress bar based on processed pages"""
        if self.total_pages.get() > 0:
            percent = int((self.current_page.get() / self.total_pages.get()) * 100)
            self.progress_percent.set(percent)
            self.progress_label.config(text=f"{percent}%")
            # Force UI update
            self.progress_bar.update()
            self.progress_label.update()
            self.root.update_idletasks()
            self.output_queue.put(("info", f"Progress updated: Page {self.current_page.get()}/{self.total_pages.get()} = {percent}%"))
    
    def check_output_queue(self):
        """Check output queue and update UI"""
        try:
            while True:
                tag, message = self.output_queue.get_nowait()
                if tag == "parse":
                    self.parse_output(message)
                elif tag == "progress":
                    # Update progress bar in main thread
                    self.progress_percent.set(message)
                    self.progress_label.config(text=f"{message}%")
                    self.progress_bar['value'] = message
                    self.progress_bar.update()
                    self.root.update_idletasks()
                elif tag == "page_progress":
                    # Update page progress in main thread
                    self.update_page_progress()
                else:
                    self.log(message, tag)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_output_queue)
    
    def update_timer(self):
        """Update processing time"""
        if self.is_processing and self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.processing_time.set(f"{elapsed}s")
            self.root.after(1000, self.update_timer)
    
    def update_page_label(self):
        self.page_label.config(text=f"{self.current_page.get()} / {self.total_pages.get()}")
    
    def update_formula_label(self):
        count = self.processed_formulas.get()
        self.formula_label.config(text=f"{count} formulas")
    
    def update_page_progress(self):
        """Update page-specific formula progress"""
        processed = self.current_page_processed.get()
        total = self.current_page_formulas.get()
        
        # Update label
        self.page_progress_label.config(text=f"{processed}/{total}")
        
        # Update progress bar
        if total > 0:
            percent = int((processed / total) * 100)
            self.page_progress_bar['value'] = percent
            self.page_progress_percent.set(percent)
        else:
            self.page_progress_bar['value'] = 0
            self.page_progress_percent.set(0)
    
    def stop_processing(self):
        if self.process:
            self.process.terminate()
            self.log("Processing stopped", 'warning')
            self.status_label.config(text="Stopped")
            self.processing_finished()
    
    def processing_finished(self):
        """Reset UI after processing finished"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if self.final_html_path.get() and os.path.exists(self.final_html_path.get()):
            self.view_html_button.config(state=tk.NORMAL)
            self.open_folder_button.config(state=tk.NORMAL)
        
        self.is_processing = False
    
    def add_zoom_functionality(self, soup):
        """Add zoom controls to EACH formula card"""
        # Add zoom control CSS for individual cards
        style_tag = soup.find('style')
        if not style_tag:
            style_tag = soup.new_tag('style')
            head = soup.find('head')
            if head:
                head.append(style_tag)
        
        zoom_css = """
        /* Individual card zoom controls */
        .card-zoom-controls {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 5px;
            display: flex;
            gap: 5px;
            align-items: center;
        }
        
        .card-zoom-btn {
            width: 28px;
            height: 28px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        
        .card-zoom-btn:hover {
            background: #f0f0f0;
            border-color: #999;
        }
        
        .card-zoom-level {
            min-width: 45px;
            text-align: center;
            font-size: 12px;
            color: #333;
        }
        
        .formula-card {
            position: relative;
            overflow: hidden;
        }
        
        .formula-image {
            position: relative;
            transform-origin: center center;
            transition: transform 0.2s ease;
        }
        """
        
        if style_tag.string:
            style_tag.string = style_tag.string + "\n" + zoom_css
        else:
            style_tag.string = zoom_css
        
        # Add zoom controls to EACH formula card
        formula_cards = soup.find_all('div', class_='formula-card')
        for idx, card in enumerate(formula_cards):
            # Create unique IDs for each card
            card_id = f"card_{idx}"
            
            # Find the formula image div
            formula_img_div = card.find('div', class_='formula-image')
            if formula_img_div:
                formula_img_div['id'] = f"formula_img_{idx}"
                
                # Create zoom controls for this card
                zoom_controls = soup.new_tag('div', attrs={'class': 'card-zoom-controls'})
                
                # Zoom out button
                zoom_out_btn = soup.new_tag('button', attrs={
                    'class': 'card-zoom-btn', 
                    'onclick': f'cardZoomOut({idx})', 
                    'title': 'Zoom Out (-)'
                })
                zoom_out_btn.string = '−'
                zoom_controls.append(zoom_out_btn)
                
                # Zoom level display
                zoom_level_div = soup.new_tag('div', attrs={
                    'class': 'card-zoom-level', 
                    'id': f'zoomLevel_{idx}'
                })
                zoom_level_div.string = '100%'
                zoom_controls.append(zoom_level_div)
                
                # Zoom in button
                zoom_in_btn = soup.new_tag('button', attrs={
                    'class': 'card-zoom-btn', 
                    'onclick': f'cardZoomIn({idx})', 
                    'title': 'Zoom In (+)'
                })
                zoom_in_btn.string = '+'
                zoom_controls.append(zoom_in_btn)
                
                # Reset button
                zoom_reset_btn = soup.new_tag('button', attrs={
                    'class': 'card-zoom-btn', 
                    'onclick': f'cardZoomReset({idx})', 
                    'title': 'Reset (0)'
                })
                zoom_reset_btn.string = '⟲'
                zoom_controls.append(zoom_reset_btn)
                
                # Insert zoom controls into the card
                card.insert(0, zoom_controls)
        
        # Add zoom JavaScript for all cards
        body = soup.find('body')
        if body:
            zoom_script = soup.new_tag('script')
            zoom_script.string = """
// Individual card zoom management
const cardZooms = {};

function updateCardZoom(cardIdx, newZoom) {
    if (!cardZooms[cardIdx]) {
        cardZooms[cardIdx] = 100;
    }
    
    cardZooms[cardIdx] = Math.max(50, Math.min(300, newZoom));
    const zoomLevel = cardZooms[cardIdx];
    
    const formulaImg = document.getElementById(`formula_img_${cardIdx}`);
    const zoomDisplay = document.getElementById(`zoomLevel_${cardIdx}`);
    
    if (formulaImg) {
        formulaImg.style.transform = `scale(${zoomLevel / 100})`;
        if (zoomDisplay) {
            zoomDisplay.textContent = zoomLevel + '%';
        }
    }
}

function cardZoomIn(cardIdx) {
    const currentZoom = cardZooms[cardIdx] || 100;
    updateCardZoom(cardIdx, currentZoom + 10);
}

function cardZoomOut(cardIdx) {
    const currentZoom = cardZooms[cardIdx] || 100;
    updateCardZoom(cardIdx, currentZoom - 10);
}

function cardZoomReset(cardIdx) {
    updateCardZoom(cardIdx, 100);
}

// Initialize all cards at 100%
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.formula-card');
    cards.forEach((card, idx) => {
        cardZooms[idx] = 100;
    });
});
"""
            body.append(zoom_script)
    
    def test_progress(self):
        """Test progress bar functionality"""
        self.log("Testing progress bar...", 'info')
        
        # Test different progress values
        test_values = [0, 25, 50, 75, 100]
        for value in test_values:
            self.progress_percent.set(value)
            self.progress_label.config(text=f"{value}%")
            self.progress_bar['value'] = value
            self.progress_bar.update()
            self.root.update_idletasks()
            self.log(f"Progress set to {value}%", 'info')
            time.sleep(0.5)
        
        # Reset to 0
        self.progress_percent.set(0)
        self.progress_label.config(text="0%")
        self.progress_bar['value'] = 0
        self.root.update_idletasks()
        self.log("Progress test complete", 'success')
    
    def view_html(self):
        """Open HTML file in default browser"""
        if self.final_html_path.get() and os.path.exists(self.final_html_path.get()):
            os.startfile(self.final_html_path.get())
            self.log(f"Opening HTML: {self.final_html_path.get()}", 'info')
    
    def open_output_folder(self):
        """Open output folder in Windows Explorer"""
        output_path = self.output_folder.get()
        if os.path.exists(output_path):
            os.startfile(output_path)
            self.log(f"Opening output folder: {output_path}", 'info')


def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = SmartNougatGUI(root)
    
    # Set window icon if available
    try:
        root.iconbitmap('smartnougat.ico')
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()