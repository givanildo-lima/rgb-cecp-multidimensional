import pandas as pd
import os

def consolidate_results(input_folder, output_folder):
    """
    Lê planilhas de resultados de experimentos por imagem e as consolida por tipo de ruído e abordagem,
    formatando os dados para o modelo de saída esperado (mv_chroma_noise.xlsx, unv_chroma_noise.xlsx).

    Args:
        input_folder (str): Caminho para a pasta contendo as planilhas de resultados por imagem.
        output_folder (str): Caminho para a pasta onde as planilhas consolidadas por ruído serão salvas.
    """
    
    # Garante que a pasta de saída exista
    os.makedirs(output_folder, exist_ok=True)

    # Dicionário para armazenar os dados, estruturado como:
    # { (tipo_ruido, abordagem): [lista_de_dataframes_para_este_grupo] }
    all_data_grouped = {}

    # Obter todos os arquivos .xlsx na pasta de entrada que seguem o padrão esperado
    input_files_in_folder = [f for f in os.listdir(input_folder) if f.startswith("resultados_") and f.endswith(".xlsx")]

    if not input_files_in_folder:
        print(f"Nenhum arquivo de resultados encontrado na pasta: {input_folder}")
        return

    for filename in input_files_in_folder:
        file_path = os.path.join(input_folder, filename)
        print(f"Processando arquivo: {filename}")

        try:
            df_image = pd.read_excel(file_path, engine='openpyxl')
            
            # Inferir a abordagem (Multivariada/Univariada) e o nome da imagem do nome do arquivo
            abordagem = "Univariada" if "unv_" in filename else "Multivariada"
            
            # Extrai o nome da imagem original do nome do arquivo
            # Remove "resultados_", o prefixo "unv_" se existir, e a extensão
            image_name_raw = filename.replace("resultados_", "").replace("unv_", "").replace("_ruidos_entropia_complexidade.xlsx", "")
            
            # Itera sobre cada Tipo_Ruido único na planilha
            for noise_type in df_image["Tipo_Ruido"].unique():
                df_subset = df_image[df_image["Tipo_Ruido"] == noise_type].copy()

                # Seleciona e renomeia as colunas para corresponder ao formato de saída desejado
                df_output_format = df_subset[["Intensidade", "Entropia", "Complexidade"]].copy()
                df_output_format["Imagem"] = image_name_raw # Adiciona o nome da imagem original como uma coluna

                group_key = (noise_type, abordagem)
                if group_key not in all_data_grouped:
                    all_data_grouped[group_key] = []
                all_data_grouped[group_key].append(df_output_format)

        except KeyError as ke:
            print(f"Erro de coluna no arquivo {filename}: {ke}. Verifique se as colunas 'Tipo_Ruido', 'Intensidade', 'Entropia', 'Complexidade' existem.")
        except Exception as e:
            print(f"Erro ao ler ou processar o arquivo {filename}: {e}")

    # Consolida e salva os resultados para cada grupo (tipo de ruído, abordagem)
    for (noise_type, abordagem), list_of_dfs in all_data_grouped.items():
        df_consolidated = pd.concat(list_of_dfs, ignore_index=True)
        
        # Cria o nome do arquivo de saída no formato desejado (mv_chroma_noise.xlsx, unv_chroma_noise.xlsx)
        # Substitui espaços e caracteres especiais para um nome de arquivo seguro
        safe_noise_name = noise_type.replace(" ", "_").replace("/", "_").replace(":", "_").lower()
        safe_abordagem_prefix = "mv" if abordagem == "Multivariada" else "unv"
        
        output_file_name = f"{safe_abordagem_prefix}_{safe_noise_name}.xlsx"
        output_file_path = os.path.join(output_folder, output_file_name)
        
        try:
            df_consolidated.to_excel(output_file_path, index=False, engine='openpyxl')
            print(f"Resultados consolidados para \'{noise_type}\' ({abordagem}) salvos em: {output_file_path}")
        except PermissionError:
            print(f"ERRO: Permissão negada ao salvar o arquivo: {output_file_path}")
            print("Por favor, verifique se o arquivo não está aberto e se você tem permissões de escrita.")
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao salvar o arquivo {output_file_name}: {e}")

if __name__ == "__main__":
    # --- CONFIGURAÇÃO --- #
    # Substitua pelo caminho real da sua pasta de resultados de entrada
    input_results_folder = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/[NOVO] Versão Organizada/V2_Experiments/"
    
    # Define a pasta de saída para os resultados consolidados
    output_consolidated_folder = os.path.join(input_results_folder, "results_noises")
    
    consolidate_results(input_results_folder, output_consolidated_folder)