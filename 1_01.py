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
