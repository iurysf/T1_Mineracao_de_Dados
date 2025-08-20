import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import joblib
import json
import os
import random

class CreateToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget; self.text = text; self.delay = delay; self.tooltip_window = None; self.schedule_id = None
        self.widget.bind("<Enter>", self.enter); self.widget.bind("<Leave>", self.leave); self.widget.bind("<ButtonPress>", self.leave)
    def enter(self, event=None): self.schedule()
    def leave(self, event=None): self.unschedule(); self.hide_tooltip()
    def schedule(self): self.unschedule(); self.schedule_id = self.widget.after(self.delay, self.show_tooltip)
    def unschedule(self):
        sid = self.schedule_id; self.schedule_id = None
        if sid: self.widget.after_cancel(sid)
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert"); x += self.widget.winfo_rootx() + 25; y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget); self.tooltip_window.wm_overrideredirect(True); self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, justify='left', background="#ffffe0", relief='solid', borderwidth=1, font=("Segoe UI", "8", "normal")); label.pack(ipadx=1)
    def hide_tooltip(self):
        tooltip = self.tooltip_window; self.tooltip_window = None
        if tooltip: tooltip.destroy()
class EntryWithPlaceholder(ttk.Frame):
    def __init__(self, master, placeholder, width=None, **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = 'grey'

        # O Entry real, onde o usuário digita
        self.entry = ttk.Entry(self, width=width)
        self.entry.pack(fill='both', expand=True) # O Entry define o tamanho do frame

        # O Label que serve como placeholder
        self.placeholder_label = ttk.Label(
            self, 
            text=self.placeholder, 
            foreground=self.placeholder_color, 
            background="#FFFFFF", 
            anchor='w',
            # --- MUDANÇA PRINCIPAL AQUI: Adiciona o cursor de texto ---
            cursor="xterm"
        )
        
        # --- MELHORIA BÔNUS: Posicionamento mais robusto ---
        # Posiciona o placeholder relativo ao Entry, não ao Frame pai
        self.placeholder_label.place(in_=self.entry, x=4, y=1, relwidth=0.975)

        # Associa os eventos
        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<FocusOut>", self._on_focus_out)
        self.placeholder_label.bind("<Button-1>", self._on_label_click)
        
        self._on_focus_out()

    def _on_focus_in(self, event=None):
        """Esconde o placeholder quando o Entry ganha foco."""
        self.placeholder_label.place_forget()

    def _on_focus_out(self, event=None):
        """Mostra o placeholder se o Entry estiver vazio ao perder o foco."""
        if not self.entry.get():
            # Usa o mesmo posicionamento robusto aqui
            self.placeholder_label.place(in_=self.entry, x=4, y=1, relwidth=0.975)

    def _on_label_click(self, event=None):
        """Dá foco ao Entry real quando o placeholder (Label) é clicado."""
        self.entry.focus()

    def get(self): return self.entry.get()
    def delete(self, first, last): self.entry.delete(first, last); self._on_focus_out()
    def focus(self): self.entry.focus()
    
# ==================== CLASSE PRINCIPAL DA APLICAÇÃO ====================
class MoviePredictorApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="arc")
        self.title("CineScope")
        self.geometry("420x780") 
        self.resizable(False, False)
        
        try:
            # Caminho para o seu ícone
            icon_path = "icons/main_icon.png"
            
            # Carrega a imagem usando Pillow...
            pil_image = Image.open(icon_path)
            # ...e a converte para um formato que o Tkinter entende
            self.app_icon = ImageTk.PhotoImage(pil_image)
            
            # Define a imagem como o ícone da janela e de suas sub-janelas
            self.iconphoto(False, self.app_icon)
            
        except FileNotFoundError:
            print(f"AVISO: Arquivo do ícone não encontrado em '{icon_path}'. O programa continuará sem ícone.")
        except Exception as e:
            print(f"AVISO: Não foi possível carregar o ícone. Erro: {e}")

        self.feature_map = {
            'budget': '💹 Orçamento',
            'votes': '👍 Votos',
            'duration': '⏱️ Duração',
            'year': '🗓️ Ano de Lançamento'
        }
        
        self._load_resources()
        self._load_icons()
        self._configure_styles()
        self._create_widgets()

    def _load_resources(self):
        try:
            base_path = "artefatos_modelo"
            self.modelos = joblib.load(os.path.join(base_path, 'todos_os_modelos.joblib'))
            self.scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
            with open(os.path.join(base_path, 'generos_lista.json'), 'r', encoding='utf-8') as f: self.generos = json.load(f)
            with open(os.path.join(base_path, 'idiomas_lista.json'), 'r', encoding='utf-8') as f: self.idiomas = json.load(f)
            with open(os.path.join(base_path, 'metricas_modelos.json'), 'r', encoding='utf-8') as f: self.metricas = json.load(f)
            with open(os.path.join(base_path, 'feature_importances.json'), 'r', encoding='utf-8') as f: self.feature_importances = json.load(f)
            with open(os.path.join(base_path, 'movie_titles.json'), 'r', encoding='utf-8') as f: self.movie_titles = json.load(f)
            with open(os.path.join(base_path, 'train_indices.json'), 'r', encoding='utf-8') as f: self.train_indices = json.load(f)
            self.colunas_modelo = self.modelos['Random Forest'].feature_names_in_
        except Exception as e:
            messagebox.showerror("Erro ao Carregar Recursos", f"Não foi possível carregar um arquivo essencial: {e}\n\nA aplicação será encerrada.")
            self.destroy()

    def _load_icons(self):
        try:
            self.icon_main = ImageTk.PhotoImage(Image.open("icons/movie_icon.png").resize((32, 32))); self.icon_predict = ImageTk.PhotoImage(Image.open("icons/predict_icon.png").resize((16, 16)))
            self.icon_clear = ImageTk.PhotoImage(Image.open("icons/clear_icon.png").resize((16, 16))); self.icon_success = ImageTk.PhotoImage(Image.open("icons/success_icon.png").resize((32, 32)))
            self.icon_failure = ImageTk.PhotoImage(Image.open("icons/failure_icon.png").resize((32, 32)))
        except Exception: self.icon_main = self.icon_predict = self.icon_clear = self.icon_success = self.icon_failure = None

    def _configure_styles(self):
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        self.style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'))
        self.style.configure('Metric.TLabel', font=('Segoe UI', 9, 'bold'))


        # Estilo para "SUCESSO"
        self.style.configure('Success.TLabel', 
                             foreground='green', 
                             font=('Segoe UI', 14, 'bold'), # Fonte maior e em negrito
                             anchor='center') # Centraliza o texto dentro do Label

        # Estilo para "FRACASSO"
        self.style.configure('Failure.TLabel', 
                             foreground='red', 
                             font=('Segoe UI', 14, 'bold'), # Fonte maior e em negrito
                             anchor='center')

        # Estilo para "Aguardando dados..."
        self.style.configure('Default.TLabel', 
                             foreground='gray', 
                             font=('Segoe UI', 14, 'italic'), # Fonte maior e em itálico
                             anchor='center')

    def _on_model_select(self, event=None):
        """Atualiza a UI (métricas e painéis visíveis) quando um modelo é selecionado."""
        nome_modelo = self.combo_modelo.get()
        if nome_modelo and nome_modelo in self.metricas:
            metricas_modelo = self.metricas[nome_modelo]; acc = metricas_modelo['accuracy']; prec = metricas_modelo['precision']; f1 = metricas_modelo['f1_score']
            self.acc_label_val.config(text=f"{acc:.3f}"); self.prec_label_val.config(text=f"{prec:.3f}"); self.f1_label_val.config(text=f"{f1:.3f}")
        
        if nome_modelo in self.feature_importances:
            self.frame_fatores.pack(pady=5, fill='x', ipady=5)
        else:
            self.frame_fatores.pack_forget()

        if nome_modelo == 'KNN':
            self.frame_similares.pack(pady=5, fill='x', ipady=5)
        else:
            self.frame_similares.pack_forget()

        self._clear_results()
        
    def _populate_random_data(self):
        """Preenche os campos numéricos com valores aleatórios dentro de faixas definidas."""
        # Definir as faixas para cada campo
        random_data = {
            'ano': random.randint(1850, 2025),
            'duração': random.randint(1, 300),
            'quantidade': random.randint(0, 4000000),
            'orçamento': random.randint(0, 1000000000)
        }

        # Iterar sobre o dicionário e preencher os campos
        for key, value in random_data.items():
            widget = self.entries[key]
            
            # Lógica para interagir com o widget EntryWithPlaceholder
            widget.placeholder_label.place_forget() # Esconde o placeholder
            widget.entry.delete(0, 'end')          # Limpa o campo
            widget.entry.insert(0, str(value))     # Insere o novo valor

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="15"); main_frame.pack(fill="both", expand=True)
        header_frame = ttk.Frame(main_frame); header_frame.pack(fill='x', pady=(0, 10)); ttk.Label(header_frame, image=self.icon_main).pack(side='left', padx=(0, 10)); ttk.Label(header_frame, text="CineScope", style='Header.TLabel').pack(side='left'); ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # --- Frames de Entradas (sem alterações) ---
        frame_numerico = ttk.LabelFrame(main_frame, text='Informações do Filme')
        frame_numerico.pack(pady=5, fill='x')
        placeholders = {'ano': 'ex: 2023', 'duração': 'ex: 120', 'quantidade': 'ex: 50000', 'orçamento': 'ex: 15000000'}
        labels_text = ['Ano de Lançamento:', 'Duração (min):', 'Quantidade de Votos:', 'Orçamento (USD):']
        self.entries = {}
        for i, texto_label in enumerate(labels_text):
            key = texto_label.split(' ')[0].lower()
            ttk.Label(frame_numerico, text=texto_label).grid(row=i, column=0, padx=5, pady=6, sticky='w')
            entry = EntryWithPlaceholder(frame_numerico, placeholder=placeholders[key], width=35)
            entry.grid(row=i, column=1, padx=5, pady=6, sticky='e')
            self.entries[key] = entry
        
        random_button = ttk.Button(
            frame_numerico, 
            text="🎲 Preencher com Dados Aleatórios", 
            command=self._populate_random_data
        )
        # Posiciona o botão na próxima linha, ocupando as duas colunas
        random_button.grid(row=4, column=0, columnspan=2, pady=(10, 5), padx=5, sticky='ew')
        
        frame_cat = ttk.LabelFrame(main_frame, text='Gênero e Idioma'); frame_cat.pack(pady=5, fill='x'); ttk.Label(frame_cat, text='Gênero Principal:').grid(row=0, column=0, padx=5, pady=6, sticky='w'); self.combo_gen1 = ttk.Combobox(frame_cat, values=self.generos, state='readonly', width=22); self.combo_gen1.grid(row=0, column=1, padx=5, pady=6, sticky='e'); ttk.Label(frame_cat, text='Idioma Original:').grid(row=1, column=0, padx=5, pady=6, sticky='w'); self.combo_idioma1 = ttk.Combobox(frame_cat, values=self.idiomas, state='readonly', width=22); self.combo_idioma1.grid(row=1, column=1, padx=5, pady=6, sticky='e')
        
        # ===== SEÇÃO DE SELEÇÃO DO MODELO (SOLUÇÃO FINAL COM PACK) =====
        frame_modelo = ttk.LabelFrame(main_frame, text="Seleção do Modelo de IA")
        frame_modelo.pack(pady=10, fill='x', padx=2)

        # Frame para a linha de seleção (label + combobox)
        selection_row_frame = ttk.Frame(frame_modelo)
        selection_row_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(selection_row_frame, text="Escolha o modelo:").pack(side='left')
        self.combo_modelo = ttk.Combobox(selection_row_frame, values=list(self.modelos.keys()), state='readonly', width=20)
        self.combo_modelo.pack(side='left', padx=5)
        self.combo_modelo.set('Random Forest'); self.combo_modelo.bind("<<ComboboxSelected>>", self._on_model_select)

        ttk.Separator(frame_modelo, orient='horizontal').pack(fill='x', pady=5, padx=5, side='top')

        # Frame principal para as métricas (que conterá as duas colunas)
        metrics_frame = ttk.Frame(frame_modelo)
        metrics_frame.pack(fill='x', expand=True, padx=10, pady=(0, 5))

        # Coluna da ESQUERDA para os rótulos de texto
        labels_col_frame = ttk.Frame(metrics_frame)
        labels_col_frame.pack(side='left')
        
        # Coluna da DIREITA para os valores das métricas
        values_col_frame = ttk.Frame(metrics_frame)
        values_col_frame.pack(side='right', fill='x', expand=True)

        # Adiciona os rótulos de texto na coluna esquerda (alinhados à esquerda)
        ttk.Label(labels_col_frame, text="Acurácia:", style='Metric.TLabel').pack(anchor='w', pady=1)
        ttk.Label(labels_col_frame, text="Precisão:", style='Metric.TLabel').pack(anchor='w', pady=1)
        ttk.Label(labels_col_frame, text="F1-Score:", style='Metric.TLabel').pack(anchor='w', pady=1)

        # Adiciona os rótulos de valor na coluna direita (alinhados à direita)
        self.acc_label_val = ttk.Label(values_col_frame, text="---", style='Metric.TLabel')
        self.acc_label_val.pack(anchor='e', pady=1)
        self.prec_label_val = ttk.Label(values_col_frame, text="---", style='Metric.TLabel')
        self.prec_label_val.pack(anchor='e', pady=1)
        self.f1_label_val = ttk.Label(values_col_frame, text="---", style='Metric.TLabel')
        self.f1_label_val.pack(anchor='e', pady=1)
        
        # ===== FIM DA SEÇÃO CORRIGIDA =====

        self.frame_fatores = ttk.LabelFrame(main_frame, text="Principais Fatores na Decisão")
        self.factor_labels = []
        # Criar 3 linhas de labels, cada uma com duas colunas
        for i in range(3):
            # Label para o número/ícone (coluna 0)
            num_label = ttk.Label(self.frame_fatores, text=f"{i+1}. ...", anchor='w')
            num_label.grid(row=i, column=0, padx=(10, 5), pady=1, sticky='w')
            
            # Label para o nome do fator (coluna 1)
            name_label = ttk.Label(self.frame_fatores, text="", anchor='w')
            name_label.grid(row=i, column=1, padx=(0, 10), pady=1, sticky='w')
            
            self.factor_labels.append((num_label, name_label))
            
        self.frame_similares = ttk.LabelFrame(main_frame, text="Filmes Similares Encontrados (apenas KNN)"); self.similar_labels = []
        for i in range(3): label = ttk.Label(self.frame_similares, text=f"🎬 ...", anchor='w'); label.pack(fill='x', padx=10, pady=1); self.similar_labels.append(label)
        
        # Frame de Resultado e Botões (já corrigido na etapa anterior)
        frame_resultado = ttk.LabelFrame(main_frame, text="Resultado da Previsão"); frame_resultado.pack(pady=5, fill='x', ipady=5)
        inner_result_frame = ttk.Frame(frame_resultado); inner_result_frame.pack(pady=10)
        self.result_icon_label = ttk.Label(inner_result_frame); self.result_text_label = ttk.Label(inner_result_frame, text="Aguardando dados...", style='Default.TLabel')
        self.result_icon_label.pack(side='left', padx=(0, 5)); self.result_text_label.pack(side='left')
        frame_botoes = ttk.Frame(frame_resultado); frame_botoes.pack(pady=(5, 10), padx=10, fill='x', expand=True)
        self.clear_button = ttk.Button(frame_botoes, text="Limpar", image=self.icon_clear, compound="left", command=self._clear_fields); self.clear_button.pack(side='right', padx=(5, 0), fill='x', expand=True)
        self.predict_button = ttk.Button(frame_botoes, text="Prever Sucesso", image=self.icon_predict, compound="left", command=self._predict); self.predict_button.pack(side='left', padx=(0, 5), fill='x', expand=True)
        
        self._on_model_select()

    def _clear_results(self):
        """Limpa todos os painéis de resultados e análises."""
        self.result_text_label.config(text="Aguardando dados...", style='Default.TLabel')
        self.result_icon_label.config(image='')
        
        # --- CORREÇÃO AQUI ---
        # Agora desempacota a tupla para acessar cada label individualmente
        for i, (num_label, name_label) in enumerate(self.factor_labels):
            num_label.config(text=f"{i+1}. ...")
            name_label.config(text="") # Limpa também o nome do fator
        
        # A limpeza dos filmes similares já estava correta
        for i, label in enumerate(self.similar_labels):
            label.config(text=f"🎬 ...")

    def _clear_fields(self):
        for widget in self.entries.values(): widget.delete(0, 'end');
        self.combo_gen1.set(''); self.combo_idioma1.set('');
        self._clear_results(); self.predict_button.focus()

    def _validate_inputs(self):
        campos_a_checar = {'Ano de Lançamento': self.entries['ano'], 'Duração': self.entries['duração'], 'Quantidade de Votos': self.entries['quantidade'], 'Orçamento': self.entries['orçamento']};
        for nome_campo, widget in campos_a_checar.items():
            if not widget.get().strip(): messagebox.showwarning('Campo Obrigatório', f'O campo "{nome_campo}" não pode estar vazio.'); widget.focus(); return None
        try:
            inputs = {}; year = float(self.entries['ano'].get());
            if not (1800 <= year <= 2100): messagebox.showerror('Erro de Validação', "O ano deve ser entre 1800 e 2100."); self.entries['ano'].focus(); return None
            inputs['year'] = year; duration = float(self.entries['duração'].get());
            if duration <= 0: messagebox.showerror('Erro de Validação', "A duração deve ser um número positivo."); self.entries['duração'].focus(); return None
            inputs['duration'] = duration; votes = float(self.entries['quantidade'].get());
            if votes < 0: messagebox.showerror('Erro de Validação', "A quantidade de votos não pode ser negativa."); self.entries['quantidade'].focus(); return None
            inputs['votes'] = votes; budget = float(self.entries['orçamento'].get());
            if budget < 0: messagebox.showerror('Erro de Validação', "O orçamento não pode ser negativo."); self.entries['orçamento'].focus(); return None
            inputs['budget'] = budget; inputs['genero'] = self.combo_gen1.get(); inputs['idioma'] = self.combo_idioma1.get();
            if not inputs['genero']: messagebox.showwarning('Campo Obrigatório', 'Por favor, selecione um gênero.'); self.combo_gen1.focus(); return None
            if not inputs['idioma']: messagebox.showwarning('Campo Obrigatório', 'Por favor, selecione um idioma.'); self.combo_idioma1.focus(); return None
            return inputs
        except ValueError: messagebox.showerror('Erro de Entrada', 'Por favor, verifique os campos numéricos.\nEles devem conter apenas números válidos.'); return None

    def _display_factors(self, user_inputs, nome_modelo):
        # Limpar os labels antes de preencher
        for num_label, name_label in self.factor_labels:
            num_label.config(text="...")
            name_label.config(text="")
            
        if nome_modelo not in self.feature_importances:
            self.factor_labels[0][0].config(text="1.")
            self.factor_labels[0][1].config(text="Análise de fatores não disponível.")
            return

        importances_modelo = self.feature_importances[nome_modelo]
        user_features = {}
        for key in ['year', 'duration', 'votes', 'budget']:
            friendly_name = self.feature_map.get(key, key.capitalize())
            user_features[friendly_name] = importances_modelo.get(key, 0)
        
        genero_nome = f"🎬 Gênero: {user_inputs['genero']}"
        idioma_nome = f"🌐 Idioma: {user_inputs['idioma']}"
        user_features[genero_nome] = importances_modelo.get(user_inputs['genero'], 0)
        user_features[idioma_nome] = importances_modelo.get(user_inputs['idioma'], 0)
        
        sorted_features = sorted(user_features.items(), key=lambda item: item[1], reverse=True)
        
        # Atualiza os labels separados
        for i, (feature_name, importance) in enumerate(sorted_features[:3]):
            num_label, name_label = self.factor_labels[i]
            
            # Divide o nome (ex: "🗓️ Ano de Lançamento") em ícone e texto
            parts = feature_name.split(" ", 1)
            icon = parts[0]
            name = parts[1] if len(parts) > 1 else ""

            num_label.config(text=f"{i+1}. {icon}")
            name_label.config(text=name)

    def _display_similar_movies(self, df_input, modelo_knn):
        try:
            _, indices = modelo_knn.kneighbors(df_input, n_neighbors=3)
            original_indices = [self.train_indices[i] for i in indices[0]]
            for i, original_idx in enumerate(original_indices):
                title = self.movie_titles.get(str(original_idx), "Título não encontrado")
                # Adiciona o emoji antes do título
                self.similar_labels[i].config(text=f"🎬 {title}")
        except Exception as e:
            for i, label in enumerate(self.similar_labels): label.config(text=f"{i+1}. Erro ao buscar")
            print(f"Erro na busca de similares: {e}")

    def _predict(self):
        user_inputs = self._validate_inputs();
        if user_inputs is None: return
        nome_modelo_escolhido = self.combo_modelo.get();
        if not nome_modelo_escolhido: messagebox.showwarning('Seleção de Modelo', 'Por favor, escolha um modelo de IA antes de prever.'); return
        
        self._clear_results()

        modelo_a_usar = self.modelos[nome_modelo_escolhido]
        
        try:
            numeric_cols = ['year', 'duration', 'votes', 'budget']; dtypes = {col: 'float64' for col in self.colunas_modelo if col in numeric_cols}
            novo_df = pd.DataFrame(0, index=[0], columns=self.colunas_modelo).astype(dtypes)
            for col in numeric_cols:
                if col in novo_df.columns: novo_df.loc[0, col] = user_inputs[col]
            if user_inputs['genero'] in novo_df.columns: novo_df.loc[0, user_inputs['genero']] = 1
            if user_inputs['idioma'] in novo_df.columns: novo_df.loc[0, user_inputs['idioma']] = 1
            
            colunas_existentes_no_df = [c for c in numeric_cols if c in novo_df.columns]
            if colunas_existentes_no_df: novo_df.loc[:, colunas_existentes_no_df] = self.scaler.transform(novo_df[colunas_existentes_no_df])
            
            resultado = modelo_a_usar.predict(novo_df)[0]
            
            if resultado == 1:
                self.result_text_label.config(text="PREVISÃO: SUCESSO", style='Success.TLabel')
                self.result_icon_label.config(image=self.icon_success)
            else:
                self.result_text_label.config(text="PREVISÃO: FRACASSO", style='Failure.TLabel')
                self.result_icon_label.config(image=self.icon_failure)
            
            if nome_modelo_escolhido in self.feature_importances: self._display_factors(user_inputs, nome_modelo_escolhido)
            if nome_modelo_escolhido == 'KNN': self._display_similar_movies(novo_df, modelo_a_usar)

        except Exception as e:
            messagebox.showerror('Erro Inesperado na Previsão', f'Ocorreu um erro durante a previsão: {e}')
            self._clear_results()

if __name__ == "__main__":
    app = MoviePredictorApp()
    app.mainloop()