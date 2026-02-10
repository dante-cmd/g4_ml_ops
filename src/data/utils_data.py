"""Utils fir etl.py"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np


def get_n_lags(periodo: int, n: int):
    """
    Obtiene el periodo n meses antes del periodo dado.
    Ejemplo:
        get_n_lags(202306, 3) -> 202303
    """
    periodo_date = datetime.strptime(str(periodo), '%Y%m')
    return int((periodo_date - relativedelta(months=n)).strftime('%Y%m'))


def get_last_n_periodos(periodo: int, n: int):
    """
    Obtiene los n periodos anteriores al periodo dado.
    Ejemplo:
        get_last_n_periodos(202306, 3) -> [202305, 202304, 202303]
    """
    return [get_n_lags(periodo, lag) for lag in range(n)]


def get_all_periodos(periodo: int):
    """
    Obtiene todos los periodos desde 2022-11 hasta el periodo dado.
    Ejemplo:
        get_all_periodos(202306) -> [202211, 202212, 202301, 202302, 202303, 202304, 202305, 202306]
    """
    periodos = pd.period_range(
            start=datetime(2022, 11, 1),
            end=datetime(periodo // 100, periodo % 100, 1),
            freq='M')
    periodos = periodos.strftime('%Y%m').astype('int32')
    return periodos


def validate_periodos(n_periodos:int|None, ult_periodo:int, all_periodos:bool):
    """
    Valida los periodos a entrenar
    """
    if n_periodos is not None:
        periodos = get_last_n_periodos(ult_periodo, n_periodos)
        min_periodo = int(np.min(periodos))
        print("Rango de actualización:",  min_periodo, "-", ult_periodo)
        # print(f"Prog Acad se está actualizando desde {min_periodo} hasta {ult_periodo}")
    else:
        assert all_periodos
        periodos = get_all_periodos(ult_periodo).copy()
        min_periodo = int(np.min(periodos))
        print("Rango de actualización:",  min_periodo, "-", ult_periodo)
        # print(f"Prog Acad se está actualizando desde {min_periodo} hasta {ult_periodo}")
    return periodos

