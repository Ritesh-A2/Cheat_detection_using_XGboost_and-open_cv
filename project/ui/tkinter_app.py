import csv
import os
import queue
import threading

from datetime import datetime

import cv2
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageTk

from video.reader import VideoReader
from utils.email import send_email
from utils.logging import get_logger

from config.settings import (
    MAX_VIDEOS,
    QUEUE_SIZE,
    TABLE_REFRESH_MS,
    RESULT_PUMP_MS,
    DISPLAY_SIZE,
)

logger = get_logger("gui")


class ExamGUI:

    def __init__(self, worker):

        self.worker = worker

        # --------------------------------------------------
        # ROOT
        # --------------------------------------------------

        self.root = tk.Tk()
        self.root.title("Exam Cheat Detection System — Proctoring Dashboard")
        self.root.geometry("1700x950")
        self.root.minsize(1300, 760)
        self.root.configure(bg="#0b0d12")

        # --------------------------------------------------
        # SESSION
        # --------------------------------------------------

        self.running = False
        self.stop_event = threading.Event()

        # --------------------------------------------------
        # VIDEO
        # --------------------------------------------------

        self.video_sources = [None for _ in range(MAX_VIDEOS)]

        self.frame_queues = [
            queue.Queue(maxsize=QUEUE_SIZE) for _ in range(MAX_VIDEOS)
        ]

        self.readers = []
        self.reader_threads = []
        self.inference_thread = None

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        self.result_queue = queue.Queue()

        # --------------------------------------------------
        # STUDENT RECORDS
        # slot -> track_id -> {roll, name, email, attempts, score, status}
        # --------------------------------------------------

        self.records = [{} for _ in range(MAX_VIDEOS)]
        self.logs = [[] for _ in range(MAX_VIDEOS)]

        # --------------------------------------------------
        # UI REFERENCES
        # --------------------------------------------------

        self.video_labels = []
        self.tables = []
        self.log_widgets = []
        self.photo_refs = [None for _ in range(MAX_VIDEOS)]
        self.source_vars = []

        self._build_ui()

        self.root.after(RESULT_PUMP_MS, self._pump_results)
        self.root.after(TABLE_REFRESH_MS, self._pump_tables)

    # ======================================================
    # UI
    # ======================================================

    def _build_ui(self):

        BG = "#0b0d12"
        PANEL = "#13161f"
        CARD = "#1a1e2a"
        BORDER = "#252c3d"
        ACC = "#e63946"
        BLUE = "#3a86ff"
        GREEN = "#2d6a4f"
        TEXT = "#e8edf4"
        MUTED = "#7a8499"

        header = tk.Frame(self.root, bg=ACC, height=55)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="◈  EXAM CHEAT DETECTION SYSTEM",
            font=("Courier New", 16, "bold"),
            bg=ACC,
            fg=TEXT,
        ).pack(side="left", padx=20)

        tk.Label(
            header,
            text="YOLO-POSE + XGBOOST  |  MULTI-VIDEO  |  GPU",
            font=("Courier New", 8),
            bg=ACC,
            fg="#ffd0d4",
        ).pack(side="right", padx=20)

        footer = tk.Frame(self.root, bg=BORDER, height=48)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)

        self.status_var = tk.StringVar(value="⏸ Idle — select videos and press START")

        tk.Label(
            footer,
            textvariable=self.status_var,
            bg=BORDER,
            fg=BLUE,
            font=("Courier New", 8),
            anchor="w",
        ).pack(side="left", padx=15)

        tk.Button(
            footer,
            text="💾 SAVE CSV",
            command=self.export_csv,
            bg=GREEN,
            fg=TEXT,
            relief="flat",
            font=("Courier New", 9, "bold"),
            padx=15,
        ).pack(side="right", padx=8)

        tk.Button(
            footer,
            text="📧 SAVE & EMAIL",
            command=self.save_and_email,
            bg=BLUE,
            fg=TEXT,
            relief="flat",
            font=("Courier New", 9, "bold"),
            padx=15,
        ).pack(side="right", padx=2)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=430)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_video_panel(body)
        self._build_right_panel(body)

    # ======================================================
    # LEFT PANEL
    # ======================================================

    def _build_left_panel(self, parent):

        BG = "#0b0d12"
        PANEL = "#13161f"
        CARD = "#1a1e2a"
        BORDER = "#252c3d"
        BLUE = "#3a86ff"
        ACC = "#e63946"
        TEXT = "#e8edf4"
        MUTED = "#7a8499"

        outer = tk.Frame(parent, bg=BG, width=300)
        outer.grid(row=0, column=0, sticky="ns", padx=(8, 4), pady=8)
        outer.grid_propagate(False)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        left = tk.Frame(canvas, bg=BG)
        window = canvas.create_window((0, 0), window=left, anchor="nw")

        left.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window, width=e.width),
        )

        def section(title):
            frame = tk.LabelFrame(
                left,
                text=f"  {title}  ",
                bg=PANEL,
                fg=MUTED,
                bd=0,
                highlightbackground=BORDER,
                highlightthickness=1,
                font=("Courier New", 9),
            )
            frame.pack(fill="x", padx=4, pady=4)
            return frame

        # VIDEO SOURCES
        source_frame = section("VIDEO SOURCES")

        for slot in range(MAX_VIDEOS):
            row = tk.Frame(source_frame, bg=PANEL)
            row.pack(fill="x", padx=8, pady=4)

            tk.Label(
                row,
                text=f"VIDEO {slot + 1}",
                bg=PANEL,
                fg=MUTED,
                width=9,
                anchor="w",
                font=("Courier New", 8),
            ).pack(side="left")

            var = tk.StringVar()
            self.source_vars.append(var)

            entry = tk.Entry(
                row,
                textvariable=var,
                bg="#1e2432",
                fg=TEXT,
                insertbackground=TEXT,
                relief="flat",
                font=("Courier New", 7),
            )
            entry.pack(side="left", fill="x", expand=True)

            tk.Button(
                row,
                text="…",
                command=lambda s=slot: self.choose_video(s),
                bg=BLUE,
                fg=TEXT,
                relief="flat",
            ).pack(side="left", padx=(4, 0))

        tk.Label(
            source_frame,
            text=(
                "4 independent readers\n"
                "1 shared YOLO-Pose model (GPU batched)\n"
                "1 batched XGBoost call per cycle"
            ),
            bg=PANEL,
            fg=MUTED,
            font=("Courier New", 7),
            justify="left",
        ).pack(anchor="w", padx=10, pady=6)

        # CONTROLS
        controls = section("CONTROLS")

        tk.Button(
            controls,
            text="▶ START EXAM",
            command=self.start_exam,
            bg=ACC,
            fg=TEXT,
            relief="flat",
            font=("Courier New", 9, "bold"),
            pady=7,
        ).pack(fill="x", padx=10, pady=5)

        tk.Button(
            controls,
            text="⏹ STOP",
            command=self.stop_exam,
            bg=CARD,
            fg=TEXT,
            relief="flat",
            font=("Courier New", 9, "bold"),
            pady=7,
        ).pack(fill="x", padx=10, pady=5)

        # EMAIL
        email = section("EMAIL ALERTS")

        self.smtp_host = self._entry(email, "SMTP Host", "smtp.gmail.com")
        self.smtp_port = self._entry(email, "SMTP Port", "587")
        self.smtp_user = self._entry(email, "Sender Email")
        self.smtp_pass = self._entry(email, "App Password", show="*")
        self.alert_to = self._entry(email, "Alert Recipient")

        tk.Label(
            email,
            text="Blank recipient = no automatic email.",
            bg=PANEL,
            fg=MUTED,
            font=("Courier New", 7),
        ).pack(anchor="w", padx=10, pady=5)

        # PERSON EDITOR
        editor = section("EDIT PERSON INFO")

        tk.Label(editor, text="Video", bg=PANEL, fg=MUTED).pack(anchor="w", padx=10)

        self.edit_video = ttk.Combobox(
            editor,
            state="readonly",
            values=[f"Video {i + 1}" for i in range(MAX_VIDEOS)],
        )
        self.edit_video.current(0)
        self.edit_video.pack(fill="x", padx=10, pady=4)

        tk.Label(editor, text="Track ID", bg=PANEL, fg=MUTED).pack(anchor="w", padx=10)

        self.edit_id = ttk.Combobox(editor, state="readonly")
        self.edit_id.pack(fill="x", padx=10, pady=4)

        self.edit_roll = self._entry(editor, "Roll No")
        self.edit_name = self._entry(editor, "Name")
        self.edit_email = self._entry(editor, "Email")

        self.edit_video.bind(
            "<<ComboboxSelected>>", lambda e: self.refresh_id_selector()
        )
        self.edit_id.bind("<<ComboboxSelected>>", lambda e: self.load_person())

        tk.Button(
            editor,
            text="✔ APPLY",
            command=self.apply_person,
            bg=BLUE,
            fg=TEXT,
            relief="flat",
            font=("Courier New", 8, "bold"),
        ).pack(anchor="w", padx=10, pady=7)

    def _entry(self, parent, label, value="", show=None):
        tk.Label(
            parent, text=label, bg="#13161f", fg="#7a8499", font=("Courier New", 7)
        ).pack(anchor="w", padx=10)

        entry = tk.Entry(
            parent,
            bg="#1e2432",
            fg="#e8edf4",
            insertbackground="#e8edf4",
            relief="flat",
            font=("Courier New", 8),
            show=show,
        )
        entry.pack(fill="x", padx=10, pady=(0, 5))

        if value:
            entry.insert(0, value)

        return entry

    # ======================================================
    # VIDEO PANEL
    # ======================================================

    def _build_video_panel(self, parent):

        centre = tk.Frame(parent, bg="#0b0d12")
        centre.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)

        for i in range(2):
            centre.rowconfigure(i, weight=1)
            centre.columnconfigure(i, weight=1)

        for slot in range(MAX_VIDEOS):
            frame = tk.LabelFrame(
                centre,
                text=f"  VIDEO {slot + 1}  ",
                bg="#13161f",
                fg="#7a8499",
                font=("Courier New", 9),
                bd=0,
                highlightbackground="#252c3d",
                highlightthickness=1,
            )
            frame.grid(row=slot // 2, column=slot % 2, sticky="nsew", padx=3, pady=3)

            label = tk.Label(
                frame,
                text=f"VIDEO {slot + 1}\nNo Feed",
                bg="black",
                fg="#7a8499",
                font=("Courier New", 10),
            )
            label.pack(fill="both", expand=True)

            self.video_labels.append(label)

    # ======================================================
    # RIGHT PANEL
    # ======================================================

    def _build_right_panel(self, parent):

        right = tk.Frame(parent, bg="#0b0d12", width=430)
        right.grid(row=0, column=2, sticky="ns", padx=(4, 8), pady=8)
        right.grid_propagate(False)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        for slot in range(MAX_VIDEOS):
            tab = tk.Frame(notebook, bg="#13161f")
            notebook.add(tab, text=f"V{slot + 1}")

            table_frame = tk.LabelFrame(
                tab, text=" DETECTION TABLE ", bg="#13161f", fg="#7a8499", bd=0
            )
            table_frame.pack(fill="both", expand=True, padx=4, pady=4)

            columns = ("ID", "ROLL", "NAME", "SCORE", "ATTEMPTS", "STATUS")

            tree = ttk.Treeview(
                table_frame, columns=columns, show="headings", selectmode="browse"
            )

            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=65, anchor="center")

            tree.pack(fill="both", expand=True)

            tree.tag_configure("cheat", background="#351010", foreground="#ff6b6b")
            tree.tag_configure("normal", background="#0b2115", foreground="#69db7c")

            self.tables.append(tree)

            log_frame = tk.LabelFrame(
                tab, text=" LIVE LOG ", bg="#13161f", fg="#7a8499", bd=0
            )
            log_frame.pack(fill="both", expand=True, padx=4, pady=4)

            text = tk.Text(
                log_frame,
                height=8,
                bg="#090b0f",
                fg="#3a86ff",
                font=("Courier New", 7),
                state="disabled",
            )
            text.pack(fill="both", expand=True)

            self.log_widgets.append(text)

    # ======================================================
    # SELECT VIDEO
    # ======================================================

    def choose_video(self, slot):
        path = filedialog.askopenfilename(
            title=f"Select Video {slot + 1}",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.video_sources[slot] = path
            self.source_vars[slot].set(path)
            self.log(slot, f"Video selected: {os.path.basename(path)}")

    # ======================================================
    # START
    # ======================================================

    def start_exam(self):
        if self.running:
            return

        selected = [
            (i, source)
            for i, source in enumerate(self.video_sources)
            if source
        ]

        if not selected:
            messagebox.showwarning("No Videos", "Select at least one video.")
            return

        self.stop_event.clear()

        self.records = [{} for _ in range(MAX_VIDEOS)]
        self.logs = [[] for _ in range(MAX_VIDEOS)]
        self.frame_queues = [
            queue.Queue(maxsize=QUEUE_SIZE) for _ in range(MAX_VIDEOS)
        ]

        self.running = True

        self.readers = []
        self.reader_threads = []

        for slot, source in selected:
            reader = VideoReader(
                slot,
                source,
                self.frame_queues[slot],
                self.stop_event,
                display_size=DISPLAY_SIZE,
            )

            thread = threading.Thread(
                target=reader.run, daemon=True, name=f"VideoReader-{slot + 1}"
            )

            self.readers.append(reader)
            self.reader_threads.append(thread)

            thread.start()
            self.log(slot, "▶ Reader started.")

        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="Central-Inference"
        )
        self.inference_thread.start()

        self.status_var.set(f"🟢 Running {len(selected)} video stream(s)")

    # ======================================================
    # INFERENCE LOOP
    # ======================================================

    def _inference_loop(self):

        active_slots = {
            slot for slot, source in enumerate(self.video_sources) if source
        }

        ended = set()

        while not self.stop_event.is_set():

            batch = []

            for slot in active_slots:
                if slot in ended:
                    continue

                try:
                    item = self.frame_queues[slot].get(timeout=0.01)
                except queue.Empty:
                    continue

                if isinstance(item, tuple) and item[0] == "END":
                    ended.add(slot)
                    continue

                batch.append(item)

            if not batch:
                if ended >= active_slots:
                    break
                continue

            try:
                results = self.worker.process_batch(batch)
                for result in results:
                    self.result_queue.put(result)

            except Exception as e:
                logger.exception("Inference batch failed")
                self.result_queue.put({"error": str(e)})

        self.root.after(0, self._processing_finished)

    # ======================================================
    # RESULT PUMP
    # ======================================================

    def _pump_results(self):
        try:
            while True:
                result = self.result_queue.get_nowait()

                if "error" in result:
                    self.status_var.set(f"❌ {result['error']}")
                    continue

                self._handle_result(result)

        except queue.Empty:
            pass

        self.root.after(RESULT_PUMP_MS, self._pump_results)

    # ======================================================
    # RESULT
    # ======================================================

    def _handle_result(self, result):

        slot = result["slot"]
        frame = result["frame"]
        tracks = result.get("tracks", [])
        events = result.get("events", [])

        for track in tracks:
            track_id = track["track_id"]

            record = self.records[slot].setdefault(
                track_id,
                {
                    "roll": str(track_id),
                    "name": f"Student {track_id}",
                    "email": "",
                    "attempts": 0,
                    "score": 0.0,
                    "status": "NORMAL",
                },
            )

            record["score"] = track.get("score", 0.0)
            record["status"] = (
                "CHEATING" if track.get("is_suspicious", False) else "NORMAL"
            )

        for event in events:
            track_id = event["track_id"]

            record = self.records[slot].setdefault(
                track_id,
                {
                    "roll": str(track_id),
                    "name": f"Student {track_id}",
                    "email": "",
                    "attempts": 0,
                    "score": event["score"],
                    "status": "CHEATING",
                },
            )

            # FIX: EventEngine already computes the authoritative
            # attempts count and puts it on the event - use it
            # directly instead of maintaining a separate counter that
            # could drift out of sync.
            record["attempts"] = event["attempts"]
            record["score"] = event["score"]
            record["status"] = "CHEATING"

            self.log(
                slot,
                (
                    f"⚠ CHEATING ATTEMPT | "
                    f"ID {track_id} | "
                    f"Attempt #{record['attempts']} | "
                    f"Score {event['score']:.2f}"
                ),
            )

            self._send_alert(slot, track_id, record, event)

        self._draw_frame(slot, frame, tracks)

    # ======================================================
    # DRAW
    # ======================================================

    def _draw_frame(self, slot, frame, tracks):

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for track in tracks:
            bbox = track.get("bbox")
            if not bbox:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            suspicious = track.get("is_suspicious", False)
            track_id = track["track_id"]
            score = track.get("score", 0.0)

            record = self.records[slot].get(track_id, {})
            attempts = record.get("attempts", 0)

            if suspicious:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"ID {track_id} | CHEAT | {score:.0%} | #{attempts}"
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {track_id} | OK | {score:.0%}"

            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - 24)),
                (min(w, x1 + 220), y1),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        self._display_frame(slot, annotated)

    # ======================================================
    # DISPLAY
    # ======================================================

    def _display_frame(self, slot, frame):

        label = self.video_labels[slot]

        width = max(label.winfo_width(), 320)
        height = max(label.winfo_height(), 180)

        h, w = frame.shape[:2]
        scale = min(width / w, height / h)

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((nw, nh), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        self.photo_refs[slot] = photo

        label.configure(image=photo, text="")

    # ======================================================
    # TABLE
    # ======================================================

    def _pump_tables(self):
        for slot in range(MAX_VIDEOS):
            self._refresh_table(slot)

        self.root.after(TABLE_REFRESH_MS, self._pump_tables)

    def _refresh_table(self, slot):

        tree = self.tables[slot]
        children = tree.get_children()

        if children:
            tree.delete(*children)

        for track_id, record in sorted(self.records[slot].items()):
            status = record["status"]
            tag = "cheat" if status == "CHEATING" else "normal"

            tree.insert(
                "",
                "end",
                values=(
                    track_id,
                    record["roll"],
                    record["name"],
                    f"{record['score']:.2f}",
                    record["attempts"],
                    "⚠ CHEAT" if status == "CHEATING" else "✓ OK",
                ),
                tags=(tag,),
            )

        self.refresh_id_selector()

    # ======================================================
    # LOG
    # ======================================================

    def log(self, slot, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = f"[{timestamp}] {message}"

        self.logs[slot].append(text)

        if len(self.logs[slot]) > 300:
            self.logs[slot] = self.logs[slot][-300:]

        self.root.after(0, lambda s=slot, m=text: self._append_log_ui(s, m))

    def _append_log_ui(self, slot, message):
        widget = self.log_widgets[slot]

        widget.configure(state="normal")
        widget.insert("end", message + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    # ======================================================
    # PERSON EDITOR
    # ======================================================

    def _selected_edit_slot(self):
        try:
            return int(self.edit_video.get().split()[-1]) - 1
        except Exception:
            return 0

    def refresh_id_selector(self):
        slot = self._selected_edit_slot()
        ids = sorted(self.records[slot].keys())
        self.edit_id["values"] = ids

    def load_person(self):
        try:
            slot = self._selected_edit_slot()
            track_id = int(self.edit_id.get())
            record = self.records[slot][track_id]
        except Exception:
            return

        self._set_entry(self.edit_roll, record["roll"])
        self._set_entry(self.edit_name, record["name"])
        self._set_entry(self.edit_email, record["email"])

    def _set_entry(self, widget, value):
        widget.delete(0, "end")
        widget.insert(0, value)

    def apply_person(self):
        try:
            slot = self._selected_edit_slot()
            track_id = int(self.edit_id.get())
        except Exception:
            messagebox.showinfo("Select Student", "Select a video and Track ID first.")
            return

        record = self.records[slot].get(track_id)
        if record is None:
            return

        record["roll"] = self.edit_roll.get().strip()
        record["name"] = self.edit_name.get().strip()
        record["email"] = self.edit_email.get().strip()

        self.log(slot, f"✏ Student {track_id} information updated.")

    # ======================================================
    # EMAIL
    # ======================================================

    def _smtp_config(self):
        user = self.smtp_user.get().strip()
        password = self.smtp_pass.get().strip()

        if not user or not password:
            return None

        try:
            port = int(self.smtp_port.get().strip())
        except Exception:
            port = 587

        return {
            "host": self.smtp_host.get().strip(),
            "port": port,
            "user": user,
            "password": password,
        }

    def _send_alert(self, slot, track_id, record, event):
        recipient = record["email"] or self.alert_to.get().strip()

        cfg = self._smtp_config()
        if not cfg or not recipient:
            return

        subject = f"Exam Alert — {record['name']}"
        body = (
            f"Cheating attempt detected.\n\n"
            f"Video: {slot + 1}\n"
            f"Track ID: {track_id}\n"
            f"Roll No: {record['roll']}\n"
            f"Student: {record['name']}\n"
            f"Attempt: {record['attempts']}\n"
            f"Score: {event['score']:.2f}\n"
        )

        threading.Thread(
            target=self._email_thread,
            args=(slot, cfg, recipient, subject, body),
            daemon=True,
        ).start()

    def _email_thread(self, slot, cfg, recipient, subject, body, attachment_path=None):
        success, error = send_email(cfg, recipient, subject, body, attachment_path)

        # FIX: original always logged email results to slot 0
        # regardless of which video the alert belonged to.
        if success:
            self.log(slot, f"📧 Alert sent to {recipient}")
        else:
            self.log(slot, f"❌ Email error: {error}")

    # ======================================================
    # CSV
    # ======================================================

    def export_csv(self):
        has_data = any(bool(records) for records in self.records)

        if not has_data:
            messagebox.showinfo("No Data", "Nothing to save yet.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"proctoring_report_{timestamp}.csv",
            filetypes=[("CSV", "*.csv")],
        )

        if not path:
            return None

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "Video",
                    "Track ID",
                    "Roll No",
                    "Name",
                    "Email",
                    "Score",
                    "Cheat Attempts",
                    "Status",
                ]
            )

            for slot in range(MAX_VIDEOS):
                for track_id, record in sorted(self.records[slot].items()):
                    writer.writerow(
                        [
                            f"Video {slot + 1}",
                            track_id,
                            record["roll"],
                            record["name"],
                            record["email"],
                            f"{record['score']:.4f}",
                            record["attempts"],
                            record["status"],
                        ]
                    )

        messagebox.showinfo("Saved", f"Report saved:\n{path}")
        return path

    # ======================================================
    # SAVE + EMAIL REPORT
    # ======================================================

    def save_and_email(self):
        path = self.export_csv()
        if not path:
            return

        cfg = self._smtp_config()
        recipient = self.alert_to.get().strip()

        if not cfg or not recipient:
            messagebox.showinfo("Email Disabled", "Configure SMTP and recipient first.")
            return

        threading.Thread(
            target=self._send_report_thread,
            args=(cfg, recipient, path),
            daemon=True,
        ).start()

        self.status_var.set("📧 Sending report...")

    def _send_report_thread(self, cfg, recipient, path):
        success, error = send_email(
            cfg,
            recipient,
            "Exam Proctoring Report",
            "Attached is the exam proctoring report.",
            attachment_path=path,
        )

        if success:
            self.root.after(0, lambda: self.status_var.set("📧 Report email sent."))
        else:
            self.root.after(0, lambda: self.status_var.set(f"❌ Email failed: {error}"))

    # ======================================================
    # STOP
    # ======================================================

    def stop_exam(self):
        if not self.running:
            return

        self.stop_event.set()
        self.running = False

        self.status_var.set("⏹ Stopping...")
        logger.info("Exam processing stopped by user.")

    # ======================================================
    # FINISHED
    # ======================================================

    def _processing_finished(self):
        if self.running:
            self.running = False
            self.status_var.set("⏹ All video streams finished.")

    # ======================================================
    # RUN
    # ======================================================

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.stop_event.set()
        self.running = False
        self.root.destroy()


def gui(worker):
    app = ExamGUI(worker)
    app.run()
