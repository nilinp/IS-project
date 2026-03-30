import plotly.graph_objects as go
import plotly.express as px
import numpy as np

DARK_BG = "#212427"
CARD_BG = "#354044"
ACCENT  = "#D45769"
ACCENT2 = "#308695"
GREEN   = "#4a9e78"
RED     = "#D45769"
GRID    = "#3d5055"
TEXT    = "#D4CFC9"
SUBTEXT = "#9aaba8"

LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font_color=TEXT,
    font_family="Space Grotesk",
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)

def confusion_matrix_fig(cm, labels=None):
    if labels is None:
        labels = ["Negative", "Positive"]
    z = cm[::-1]
    x = labels
    y = labels[::-1]
    annotations = []
    for i in range(len(z)):
        for j in range(len(z[i])):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=f"<b>{z[i][j]}</b>",
                showarrow=False,
                font=dict(size=20, color="white"),
            ))
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#1e2f3f"], [0.5, "#455054"], [1, "#F67280"]],
        showscale=False,
    ))
    fig.update_layout(**LAYOUT, title="Confusion Matrix",
                      annotations=annotations, height=320)
    return fig

def roc_curve_fig(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color=SUBTEXT, width=1),
        name="Random (AUC=0.5)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=ACCENT, width=3),
        name=f"ROC (AUC={auc:.3f})", fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)",
    ))
    fig.update_layout(**LAYOUT, title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=350, legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0))
    return fig

def feature_importance_fig(names, importances, title="Feature Importances"):
    idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0,"#455054"],[0.5,ACCENT],[1,ACCENT2]],
        ),
    ))
    fig.update_layout(**LAYOUT, title=title, xaxis_title="Importance", height=420)
    return fig

def training_history_fig(history, metric="accuracy"):
    epochs = list(range(1, len(history[metric])+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history[metric], mode="lines",
        name="Train", line=dict(color=ACCENT, width=2),
    ))
    val_key = f"val_{metric}"
    if val_key in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history[val_key], mode="lines",
            name="Validation", line=dict(color=GREEN, width=2),
        ))
    label = metric.replace("_", " ").title()
    fig.update_layout(**LAYOUT, title=f"Training {label}", xaxis_title="Epoch", yaxis_title=label, height=330, legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0))
    return fig

def distribution_fig(df, col, color_col=None, title=None):
    if color_col:
        fig = px.histogram(df, x=col, color=color_col, color_discrete_sequence=[ACCENT, GREEN], barmode="overlay", opacity=0.75)
    else:
        fig = px.histogram(df, x=col, color_discrete_sequence=[ACCENT])
    fig.update_layout(**LAYOUT, title=title or col, height=300)
    return fig

def correlation_fig(df):
    corr = df.corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(**LAYOUT, title="Correlation Matrix", height=520, margin=dict(l=80, r=20, t=60, b=80))
    return fig
