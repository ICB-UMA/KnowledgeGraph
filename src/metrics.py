def calculate_topk_accuracy(df, topk_values):
    """
    Calcula el Top-k accuracy para cada valor de k en topk_values.
    
    :param df: DataFrame que contiene las columnas 'code' y 'codes'.
    :param topk_values: Lista de valores de k para calcular el Top-k accuracy.
    :return: Diccionario con los valores de k como claves y las precisiones como valores.
    """
    # Inicializar diccionario para almacenar los resultados
    topk_accuracies = {k: 0 for k in topk_values}

    for index, row in df.iterrows():
        true_code = row['code']
        predicted_codes = row['codes']
        seen = set()
        unique_candidates = [x for x in predicted_codes if not (x in seen or seen.add(x))]

        for k in topk_values:
            if true_code in unique_candidates[:k]:
                topk_accuracies[k] += 1

    total_rows = len(df)
    for k in topk_values:
        topk_accuracies[k] = topk_accuracies[k] / total_rows

    return topk_accuracies