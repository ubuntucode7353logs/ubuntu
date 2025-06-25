import csv
import json

import joblib
import pandas as pd
import datetime

import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc

from transformers import AutoTokenizer, AutoModel
from dash.exceptions import PreventUpdate

from classifier import CustomEmbedder
from question_similarity_finder import QuestionSimilarityFinderTfidf
from utils.utils import is_personal_request, has_personal_verbs, anonymization

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

alfa_base = pd.read_excel('./data/alfa_base.xlsx', engine='openpyxl')
with open('./data/personal_answers.json', 'r', encoding='utf-8') as f:
    personal_answers_base = json.load(f)

model_path = "./models/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

finder_tfidf = QuestionSimilarityFinderTfidf(model, tokenizer)
finder_tfidf.prepare_question_base(alfa_base)

loaded_tokenizer_path = "./models/logistic_regression"
loaded_model_path = "./models/logistic_regression"
loaded_clf_path = "./models/logistic_regression/logistic_regression_model.pkl"

embedder = CustomEmbedder(loaded_tokenizer_path)
clf = joblib.load(loaded_clf_path)


navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Суфлёр Collection", href="/", className="fw-bold fs-4 custom-navbar-brand"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Описание", href="/description", active="exact", className="custom-navlink")),
                 dbc.NavItem(dbc.NavLink("Бот", href="/dashboard", active="exact", className="custom-navlink")),],
            className="ms-auto", navbar=True),
        dbc.NavItem(
            dbc.Button("Рестарт", color="danger", id="restart-btn", className="ms-3"), className="ms-2"),
        dcc.Location(id="url", refresh=True)
    ]),
    color="light", className="mb-4 shadow-sm border-bottom")

app.layout = html.Div(
        style={"color": "#00A69A", "minHeight": "100vh"},
        children=[
            dcc.Location(id='url', refresh=False), navbar,
            dbc.Container(id='page-content', className="animate__animated animate__fadeIn", fluid=True, class_name="px-4")
        ])


description_layout = dbc.Container([
    html.H2("📘 Описание проекта", className="mb-4", style={"color": "#00A69A"}),

    dbc.Row([
        dbc.Col([
            html.P(
                "Сейчас проводится тестирование алгоритма и наполнение базы знаний. Алгоритм состоит из двух частей:",
                style={"fontSize": "16px", "color": "#00A69A"}
            ),

            html.Ul([
                html.Li("🔍 Поиск наиболее похожего вопроса из информационной справки.", style={"color": "#495867"}),
                html.Li("👤 Поиск персонализированного ответа из шаблонов (если вопрос задан от первого лица).", style={"color": "#495867"}),
            ], style={"fontSize": "15px"}),

            html.P(
                "Цель — даже при неполной базе знаний выдать хоть какую-то релевантную подсказку для ответа клиенту. Что это значит на практике?",
                style={"fontSize": "16px", "marginTop": "1rem", "color": "#00A69A"}
            ),

            dbc.Row([
                dbc.Col([
                    html.Img(src="./assets/good_answer.png", style={"width": "100%", "borderRadius": "12px",
                                                                   "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),
                    html.P("🙂 Найдена нужная справочная информация и корректный персональный ответ.",
                           className="text-center", style={"marginTop": "0.5rem", "color": "#00A69A"})
                ], md=4),

                dbc.Col([
                    html.Img(src="./assets/good_personal_answer_bad_info.png", style={"width": "100%", "borderRadius": "12px",
                                                                                     "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),
                    html.P("👤 Найден шаблон персонального ответа, но справочной информации по этому вопросу нет.",
                           className="text-center", style={"marginTop": "0.5rem", "color": "#00A69A"})
                ], md=4),

                dbc.Col([
                    html.Img(src="./assets/good_info_answer_bad_personal.png", style={"width": "100%", "borderRadius": "12px",
                                                                                     "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),
                    html.P("📚 Есть подходящая справочная информация, но нет персонального ответа на данный случай.",
                           className="text-center", style={"marginTop": "0.5rem", "color": "#00A69A"})
                ], md=4),
            ], className="my-4"),

            html.P("Например, если вы ввели:", style={"fontSize": "16px", "color": "#495867"}),
            html.Blockquote("«хочу взять кредит онлайн»", style={"fontStyle": "italic", "color": "#555"}),

            html.P(
                "Такой вопрос будет отмечен как персонализированный (то есть задан от первого лица), "
                "но в базе пока может не оказаться подходящего шаблона — тогда сработает хотя бы справочная часть.",
                style={"fontSize": "16px", "color": "#495867"}
            ),

            html.P(
                "Почему важно тестировать модели даже на неполных данных? Как и в большинстве задач машинного обучения, "
                "мы не можем заранее предсказать все формулировки, которые будут использовать клиенты. Невозможно также создать идеальную предобработку текста. "
                "Поэтому самый эффективный способ — это постепенное накопление данных в процессе практического применения.",
                style={"fontSize": "16px", "color": "#495867"}
            ),

            html.P(
                "✔️ Если хотя бы одна из моделей (поиск или персонализация) сработала — это уже хороший результат! "
                "Но если у вас есть идея, как дополнить или уточнить ответ — смело оставляйте отрицательный фидбэк с комментарием.",
                style={"fontSize": "16px", "marginTop": "1rem", "color": "#495867"}
            ),

            html.P("Спасибо за участие! 🤝", style={"fontSize": "16px", "fontWeight": "bold", "marginTop": "1rem", "color": "#00A69A"}),

            # Новый блок про заполнение фидбэка
            html.H3("Заполнение фидбэка", style={"color": "#fe5f55", "marginTop": "2rem"}),
            html.P(
                "В фидбэке вводится корректный ответ и ФИО, чтобы при необходимости можно было проконсультироваться "
                "по составлению нового ответа для базы знаний.",
                style={"fontSize": "15px", "color": "#6c757d"}
            ),
            html.P(
                "Если есть ещё вопросы, их можно присылать мне на почту ПОЧТА и в скайп/teams.",
                style={"fontSize": "15px", "color": "#6c757d"}
            ),
            html.P(
                "Примечание: модель может выдавать полностью некорректные ответы. Для этого и проводится тестирование.",
                style={"fontSize": "15px", "color": "#6c757d", "fontStyle": "italic"}
            ),

            # Фраза про сброс данных
            html.P(
                "Чтобы сбросить все введённые данные жмите Restart.",
                style={"fontSize": "15px", "color": "#6c757d", "fontStyle": "italic"}
            ),
        ])
    ])
], className="animate__animated animate__fadeIn")


main_layout = dbc.Container([
        html.Div([
                dcc.Store(id="session-store", storage_type="session"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                        html.H5("FEEDBACK", className="feedback-title"),
                                        dbc.Button("Ответ удовлетворительный", id="satisfactory-answer-btn", size="sm", className="good-feedback-button"),
                                        html.Br(),
                                        html.H5("Модель дала некорректный ответ? - заполните фидбек", className="bad-feedback-label"),
                                        dbc.Label("ФИО оператора:", className="feedback-label"),
                                        dbc.Input(id="input-operator-fio", placeholder="Введите ФИО оператора", type="text", size="sm",
                                                  className="feedback-input"),
                                        dbc.Label("Корректный ответ:", className="feedback-label"),
                                        dbc.Input(id="input-correct-answer", placeholder="Введите корректный ответ", type="text", size="sm",
                                                  className="feedback-input"),
                                        dbc.Button("✉️ Отправить", id="submit-feedback-btn", size="sm", className="bad-feedback-button"),
                                    ]),className="shadow p-3 bg-light",), width=3,),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4("База Знаний", className="knowledge-header"),
                                    html.Div([
                                        dbc.Label("Введите запрос:", className="text-secondary small-label mb-1"),
                                        dbc.InputGroup([
                                            dbc.Input(id="input-text", placeholder="Вопрос клиента...", type="text", size="sm", className="small-input"),
                                            dbc.Button("Ввод", id="submit-btn", size="sm", className="inline-button")
                                        ], className="mb-3")
                                    ]),

                                    html.Div([
                                        html.H5("Персональные данные", className="mb-2 small-label", style={"color": "#00A69A"}),
                                        html.Div([
                                            dbc.Checkbox(id="chk-fio", value=True, disabled=True, className="me-2"),
                                            dbc.Label("ФИО:", html_for="input-fio", className="me-2 small-label"),
                                            dbc.Input(id="input-fio", type="text", placeholder="Введите ФИО", size="sm", className="flex-grow-1 small-input")
                                        ], className="d-flex align-items-center mb-2"),

                                        html.Div([
                                            html.Div([
                                                dbc.Checkbox(id="chk-date", className="me-2"),
                                                dbc.Label("Дата:", html_for="input-date", className="me-2 small-label"),
                                                dcc.DatePickerSingle(id="input-date", date=datetime.date.today(), display_format="DD.MM.YYYY",
                                                                     className="flex-grow-1")
                                            ], className="d-flex align-items-center me-3", style={"flex": "1"}),
                                            html.Div([
                                                dbc.Checkbox(id="chk-phone", className="me-2"),
                                                html.Span("☎️", className="me-2 small-label"),
                                                dbc.Input(id="input-phone", type="tel", placeholder="+375 (__) ___-__-__", inputMode="tel",
                                                    pattern=r"^\+375\s\(\d{2}\)\s\d{3}-\d{2}-\d{2}$", size="sm",className="flex-grow-1 small-input"
                                                )
                                            ], className="d-flex align-items-center", style={"flex": "1"}),
                                        ], className="d-flex align-items-center mb-2"),

                                        html.Div([
                                            dbc.Checkbox(id="chk-amount", className="me-2"),
                                            dbc.Label("Сумма:", html_for="input-amount", className="me-2 small-label"),
                                            dbc.Input(id="input-amount", type="number", placeholder="Введите сумму", size="sm",
                                                      className="flex-grow-1 me-2 small-input"),
                                            dbc.Select(id="select-currency",
                                                options=[
                                                    {"label": "BYN", "value": "BYN"},
                                                    {"label": "RUB", "value": "RUB"},
                                                    {"label": "USD", "value": "USD"},
                                                    {"label": "EUR", "value": "EUR"},
                                                    {"label": "CNY", "value": "CNY"}
                                                ], value="BYN",size="sm", className="currency-select")], className="d-flex align-items-center mb-3"),

                                        dbc.Button("Ввод персональных данных", id="submit-personal", size="sm", className="primary-button")
                                    ])]),
                                className="shadow p-3 bg-light"), width=5,),

                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        html.H5("Возможный персонализированный ответ:", className="mb-3 small personal-answers-title"),
                                        *[
                                            html.Div([
                                                html.P(f"Персональный ответ {i}:", className="personal-answer-label"),
                                                html.Div(className="copy-wrapper",
                                                    children=[
                                                        dbc.Textarea(id=f"output-personal-{i}", value="", readOnly=True, className="mb-2 personal-answer-textarea"),
                                                        dbc.Button("📋", id=f"copy-btn-personal-{i}",color="light", size="sm", className="copy-button")
                                                    ])]) for i in range(1, 4)]])]),
                                className="shadow p-3 bg-light",), width=4,),], className="mb-4"),

                dbc.Row([
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                        html.Div([
                                            dbc.Label("Возможные ответы клиенту из БЗ:", className="small text-secondary label-text"),
                                            html.Div([
                                                html.P(id="label-text-1", className="mb-1 label-text"),
                                                html.Div(className="textarea-wrapper", children=[dbc.Textarea(id="output-text-1", value="", readOnly=True,
                                                                                                              className="personal-answer-textarea mb-2"),
                                                        dbc.Button("📋", id="copy-btn-1", color="light", size="sm", className="copy-button")])]),
                                            html.Div([
                                                html.P(id="label-text-2", className="mb-1 label-text"),
                                                html.Div(className="textarea-wrapper", children=[dbc.Textarea(id="output-text-2", value="", readOnly=True,
                                                                                                              className="personal-answer-textarea mb-2"),
                                                        dbc.Button("📋", id="copy-btn-2", color="light", size="sm", className="copy-button")])]),
                                            html.Div([
                                                html.P(id="label-text-3", className="mb-1 label-text"),
                                                html.Div(className="textarea-wrapper",children=[
                                                        dbc.Textarea(id="output-text-3", value="", readOnly=True, className="personal-answer-textarea mb-2"),
                                                        dbc.Button("📋", id="copy-btn-3", color="light", size="sm", className="copy-button")
                                                    ])])], className="mb-4"),]),
                                className="card-container shadow p-3 bg-light"), width=12,),], className="mb-4"),],
            className="dashboard-container"
        ),

        dbc.Toast("Скопировано!", id="copy-toast-1", header="", duration=2000, is_open=False, dismissable=False, className="copy-toast"),
        dbc.Toast("Скопировано!", id="copy-toast-2", header="", duration=2000, is_open=False, dismissable=False, className="copy-toast"),
        dbc.Toast("Скопировано!", id="copy-toast-3", header="", duration=2000, is_open=False, dismissable=False, className="copy-toast"),
        dbc.Toast("Скопировано!", id="copy-toast-personal-1", header="", duration=2000, is_open=False, dismissable=False, className="copy-toast"),
        dbc.Toast("Скопировано!", id="copy-toast-personal-2", header="", duration=2000, is_open=False, dismissable=False, className="copy-toast"),
        dbc.Toast("Скопировано!", id="copy-toast-personal-3", header="", duration=2000, is_open=False, dismissable=False, className="copy-toast"),
    ]
)
samples = [
    "Добрый день. Подскажите Ваши ФИО",
    "Выбирайте банковские, финансовые услуги - банки - Альфа-Банк - погашение кредита и вводите номер счета выше",
    "Пожалуйста, хорошего дня!",
    "По данному вопросу рекомендуем обратиться в контакт-центр банка: 198 – Единый номер по Беларуси, +375 (29) 733-33-32 – МТС, +375 (44) 733-33-32 – A1, +375 (25) 733-33-32 – life:)",
    "Чем могу помочь?",
    "Отметили информацию, не позднее 30 июня ждем оплату, сумму на момент оплаты уточните",
    "РЕФИНАСИРОВАНИЕ - если у вас возникли финансовые трудности, можем предложить рефинансирование вашего кредита. детали по номеру",
    "ОПИ - для этого необходимо открыть транзитный счет дистанционно, связаться со специалистом по номеру +375445057773, либо с паспортом обратиться в отделение банка. После того как счет будет открыт, можно оплачивать и ОПИ не спишет средства",
    "Оплата поступила, задолженность сегодня будет списана.",
    "Уточните причину не оплаты по кредиту?",
    "С учетом длительной задолженности срок для погашения не позднее…",
    "Уточните Ваш вопрос?",
    "Уточните номер телефона для связи? 11:25:24 ДП",
    "Срок для погашения не позднее.... В случае не оплаты, документы будут переданы на следующий этап взыскания, ожидаем оплату...."
]

gallery_layout = html.Div([
    html.H4("🎓 Типовые ответы", style={"margin": "20px 0"}),

    html.Div(
        [
            html.Div(
                sample,
                id={"type": "sample-text", "index": i},
                n_clicks=0,
                **{"data-text": sample},
                style={
                    "cursor": "pointer",
                    "padding": "10px",
                    "border": "1px solid #ccc",
                    "borderRadius": "5px",
                    "marginBottom": "10px",
                    "backgroundColor": "#f8f9fa"
                }
            )
            for i, sample in enumerate(samples)
        ],
        id="sample-gallery",
        style={
            "maxHeight": "80vh",
            "overflowY": "auto",
            "width": "400px",
            "padding": "10px",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
        }
    ),

    html.Div(id="copy-helper", style={"display": "none"})
])

dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col(main_layout, width=9),     # например, 9/12 ширины
        dbc.Col(gallery_layout, width=3),  # 3/12 ширины
    ])
])


app.clientside_callback(
    """
    function(n_clicks_list) {
        const triggered = dash_clientside.callback_context.triggered;
        if (!triggered.length) return window.dash_clientside.no_update;

        const id = triggered[0].prop_id.split(".")[0];
        const element = document.getElementById(id);
        if (!element) return window.dash_clientside.no_update;

        const text = element.getAttribute("data-text");
        if (!text) return window.dash_clientside.no_update;

        // Создание временного textarea
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);

        return "";
    }
    """,
    Output("copy-helper", "children"),
    Input({"type": "sample-text", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const textArea = document.getElementById('output-text-1');
        if (textArea) {
            textArea.select();
            document.execCommand('copy');
        }
        return true;
    }
    """,
    Output("copy-toast-1", "is_open"),
    Input("copy-btn-1", "n_clicks")
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const textArea = document.getElementById('output-text-2');
        if (textArea) {
            textArea.select();
            document.execCommand('copy');
        }
        return true;
    }
    """,
    Output("copy-toast-2", "is_open"),
    Input("copy-btn-2", "n_clicks")
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const textArea = document.getElementById('output-text-3');
        if (textArea) {
            textArea.select();
            document.execCommand('copy');
        }
        return true;
    }
    """,
    Output("copy-toast-3", "is_open"),
    Input("copy-btn-3", "n_clicks")
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const textArea = document.getElementById('output-personal-1');
        if (textArea) {
            textArea.select();
            document.execCommand('copy');
        }
        return true;
    }
    """,
    Output("copy-toast-personal-1", "is_open"),
    Input("copy-btn-personal-1", "n_clicks")
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const textArea = document.getElementById('output-personal-2');
        if (textArea) {
            textArea.select();
            document.execCommand('copy');
        }
        return true;
    }
    """,
    Output("copy-toast-personal-2", "is_open"),
    Input("copy-btn-personal-2", "n_clicks")
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const textArea = document.getElementById('output-personal-3');
        if (textArea) {
            textArea.select();
            document.execCommand('copy');
        }
        return true;
    }
    """,
    Output("copy-toast-personal-3", "is_open"),
    Input("copy-btn-personal-3", "n_clicks")
)

@app.callback(
    Output("url", "pathname"),
    Input("restart-btn", "n_clicks"),
    prevent_initial_call=True
)
def restart_page(n_clicks):
    return "/dashboard"


def preprocess_custom_request(request):
    is_personal = has_personal_verbs(request)
    search_base_knowledge = finder_tfidf.get_top_similar(request)
    question1, question2, question3 = search_base_knowledge['вопрос'].values.tolist()[:3]
    answer1, answer2, answer3 = search_base_knowledge['ответ'].values.tolist()[:3]
    return is_personal, answer1, answer2, answer3, question1, question2, question3


@app.callback(
    [
        Output("output-text-1", "value"),
        Output("output-text-2", "value"),
        Output("output-text-3", "value"),
        Output("label-text-1", "children"),
        Output("label-text-2", "children"),
        Output("label-text-3", "children"),
        Output("session-store", "data")# сохраняем данные в сессии
    ],
    Input("submit-btn", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True
)
def update_output(n_clicks, user_input):
    default_outputs = ("", "", "", "", "", "", {})

    if not user_input:
        return default_outputs

    is_personal, answer1, answer2, answer3, question1, question2, question3 = preprocess_custom_request(user_input)

    if not is_personal:
        is_personal = is_personal_request(user_input)

    session_data = {"is_personal": is_personal}

    return answer1, answer2, answer3, question1, question2, question3, session_data


def generate_greeting():
    current_hour = datetime.datetime.now().hour

    if 5 <= current_hour < 12:
        return "Доброе утро!"
    elif 12 <= current_hour < 18:
        return "Добрый день!"
    elif 18 <= current_hour < 23:
        return "Добрый вечер!"
    else:
        return "Здравствуйте!"


def fill_tags(template, tags):
    for key, value in tags.items():
        template = template.replace(f"[{key}]", value)
    return template



@app.callback(
    Output("output-personal-1", "value"),
    Output("output-personal-2", "value"),
    Output("output-personal-3", "value"),
    Input("submit-personal", "n_clicks"),
    State("input-fio", "value"),
    State("input-date", "date"),
    State("input-amount", "value"),
    State("select-currency", "value"),
    State("input-phone", "value"),
    State("chk-phone", "value"),
    State("input-text", "value"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def handle_personal_submit(n_clicks, fio, date, amount, currency, phone, chk_phone, user_question, session_data):
    is_personal = session_data.get("is_personal") if session_data else False
    if not is_personal or not user_question:
        return "", "", ""

    tags = {"HELLO": generate_greeting(), "FIO": fio if fio else "[FIO]",
        "ДАТА": "[ДАТА]", "СУММА": "[СУММА]", "TELSP": "[TELSP]"}

    if date:
        try:
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
            tags["ДАТА"] = date_obj.strftime('%d-%m-%Y')
        except Exception:
            pass

    if amount:
        tags["СУММА"] = f"{amount} {currency}" if currency else f"{amount}"

    if chk_phone and phone:
        tags["TELSP"] = phone
    else:
        tags["TELSP"] = session_data.get("telsp", "[TELSP]") if session_data else "[TELSP]"

    embeddings = embedder.get_embeddings(user_question)
    predictions = clf.predict(embeddings)
    templates = personal_answers_base.get(predictions[0], {})

    results = []
    for template in list(templates.values())[:3]:
        filled = fill_tags(template, tags)
        results.append(filled)

    while len(results) < 3:
        results.append("")

    print(f"Категория: {predictions[0]}, теги: {tags}")
    return results[0], results[1], results[2]



@app.callback(
    Output('satisfactory-answer-btn', 'children'),
    Input('satisfactory-answer-btn', 'n_clicks'),
    State('input-text', 'value'),
    State('output-text-1', 'value'),
    State('output-text-2', 'value'),
    State('output-text-3', 'value'),
    prevent_initial_call=True
)
def handle_satisfactory_answer(n_clicks, question, ans1, ans2, ans3):
    if not n_clicks:
        raise PreventUpdate

    if not question or not ans1 or not ans2 or not ans3:
        raise PreventUpdate

    anonymized_question = anonymization(question)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [current_time, "", anonymized_question, True, ""]

    with open('./data/feedback_logs.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return "✅ Ответ принят"


@app.callback(
    Output('submit-feedback-btn', 'children'),
    Input('submit-feedback-btn', 'n_clicks'),
    State('input-operator-fio', 'value'),
    State('input-correct-answer', 'value'),
    State('input-text', 'value'),
    State('output-text-1', 'value'),
    State('output-text-2', 'value'),
    State('output-text-3', 'value'),
    prevent_initial_call=True
)
def handle_feedback(n_clicks, operator_fio, correct_answer, question, ans1, ans2, ans3):
    if not n_clicks:
        raise PreventUpdate

    if not question or not ans1 or not ans2 or not ans3:
        raise PreventUpdate

    if operator_fio and correct_answer:
        anonymized_question = anonymization(question)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [current_time, operator_fio, anonymized_question, False, correct_answer]

        with open('./data/feedback_logs.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        return "✅ Отправлено"
    else:
        return "⚠️ Заполните все поля"



@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/dashboard':
        return dashboard_layout
    elif pathname == '/description':
        return description_layout
    return description_layout


if __name__ == '__main__':
    app.run(debug=True)
