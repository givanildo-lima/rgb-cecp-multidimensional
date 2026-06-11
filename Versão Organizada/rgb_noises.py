import numpy as np
import cv2
import warnings
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
from typing import Tuple, Optional
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

#%%   

"""
Sintetizador completo de ruídos específicos para imagens coloridas

Suporta todos os ruídos discutidos:
1. Ruído de Crominância (Chroma Noise)
2. Ruído de Desalinho de Cores (Color Misalignment)
3. Moiré de Cor (Color Moiré)
4. Ruído de Matriz de Bayer (Bayer Pattern Noise)
5. Ruído Dependente da Cor (Color-Dependent Noise)
6. Ruído de Balanceamento de Branco (White Balance Noise)
7. Ruído de Cross-Talk entre Canais
8. Ruído de Compressão Colorida (JPEG Color Artifacts)
9. Ruído de Filtro Anti-Aliasing Colorido
10. Aberração Cromática
11. Flare Colorido
12. Ruído de Demosaicking
"""
  
# ==============================
# 1. RUÍDO DE CROMINÂNCIA (CHROMA NOISE)
# ==============================

def add_chroma_noise(image: np.ndarray, 
                 noise_strength: float = 0.05,
                 correlation: float = 0.3) -> np.ndarray:
    """
    Adiciona ruído de crominância (manchas coloridas)
    
    Args:
        image: Imagem RGB [0-255]
        noise_strength: Intensidade do ruído (0-0.2)
        correlation: Correlação entre canais de crominância
    
    Returns:
        Imagem com ruído de crominância
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # Converter para Lab
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Gerar ruído correlacionado para canais a* e b*
    height, width = lab.shape[:2]
    
    # Ruído base
    noise_base = np.random.randn(height, width)
    
    # Suavizar para criar manchas (não ruído pontual)
    kernel_size = max(3, int(min(height, width) * 0.01))
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    noise_smooth = convolve2d(noise_base, kernel, mode='same', boundary='symm')
    
    # Ruído para canais a* e b* com correlação
    noise_a = noise_smooth * noise_strength * 128
    noise_b = (correlation * noise_smooth + 
              np.sqrt(1 - correlation**2) * np.random.randn(height, width)) * noise_strength * 128
    
    # Aplicar apenas aos canais de crominância
    lab[:, :, 1] = np.clip(lab[:, :, 1] + noise_a, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + noise_b, 0, 255)
    
    # Converter de volta para RGB
    noisy = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

# ==============================
# 2. RUÍDO DE DESALINHO DE CORES
# ==============================

def add_color_misalignment(image: np.ndarray,
                      shift_range: Tuple[float, float] = (0.5, 2.0),
                      rotation_range: Tuple[float, float] = (-0.5, 0.5)) -> np.ndarray:
    """
    Desalinha os canais de cor (simula registro imperfeito)
    
    Args:
        image: Imagem RGB
        shift_range: Alcance do deslocamento em pixels
        rotation_range: Alcance da rotação em graus
    
    Returns:
        Imagem com canais desalinhados
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Separar canais
    channels = cv2.split(image)
    
    # Aplicar transformações diferentes a cada canal
    noisy_channels = []
    
    for i, channel in enumerate(channels):
        # Parâmetros aleatórios para cada canal
        dx = np.random.uniform(-shift_range[1], shift_range[1])
        dy = np.random.uniform(-shift_range[1], shift_range[1])
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        
        # Matriz de transformação
        M = cv2.getRotationMatrix2D((channel.shape[1]/2, channel.shape[0]/2), angle, 1)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # Aplicar transformação afim
        transformed = cv2.warpAffine(channel, M, (channel.shape[1], channel.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        noisy_channels.append(transformed)
    
    # Combinar canais
    misaligned = cv2.merge(noisy_channels)
    return np.clip(misaligned, 0, 255).astype(np.uint8)

# ==============================
# 3. MOIRÉ DE COR
# ==============================

def add_color_moire(image: np.ndarray,
               frequency_range: Tuple[float, float] = (0.05, 0.2),
               amplitude_range: Tuple[float, float] = (0.1, 0.3)) -> np.ndarray:
    """
    Adiciona padrões de moiré colorido
    
    Args:
        image: Imagem RGB
        frequency_range: Faixa de frequência do padrão
        amplitude_range: Amplitude do padrão
    
    Returns:
        Imagem com moiré
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    height, width = image.shape[:2]
    
    # Criar padrões senoidais para cada canal
    moire_pattern = np.zeros_like(image)
    
    for c in range(3):
        # Frequência e direção aleatórias para cada canal
        freq = np.random.uniform(frequency_range[0], frequency_range[1])
        angle = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        
        # Criar grade de coordenadas
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Padrão senoidal
        pattern = np.sin(2 * np.pi * freq * (X * np.cos(angle) + Y * np.sin(angle)))
        
        # Adicionar harmônicos para padrão mais complexo
        for harmonic in range(2, 4):
            freq_h = freq * harmonic
            pattern += 0.5/harmonic * np.sin(2 * np.pi * freq_h * 
                                            (X * np.cos(angle) + Y * np.sin(angle)))
        
        moire_pattern[:, :, c] = pattern * amplitude
    
    # Aplicar padrão (mais forte em áreas de alta frequência)
    edges = cv2.Canny((image.mean(axis=2) * 255).astype(np.uint8), 50, 150) / 255.0
    edge_mask = ndimage.gaussian_filter(edges, sigma=3)
    
    moire_weighted = moire_pattern * edge_mask[:, :, np.newaxis]
    
    noisy_image = image + moire_weighted
    
    return np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

# ==============================
# 4. RUÍDO DE MATRIZ DE BAYER
# ==============================

def simulate_bayer_noise(image: np.ndarray,
                    noise_levels: Tuple[float, float, float] = (0.02, 0.01, 0.03)) -> np.ndarray:
    """
    Simula ruído específico do padrão Bayer
    
    Args:
        image: Imagem RGB
        noise_levels: Níveis de ruído para (R, G, B)
    
    Returns:
        Imagem com ruído Bayer
    """
    
    bayer_pattern = np.array([[0, 1], [1, 2]])  # RGB
    
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    height, width = image.shape[:2]
    
    # Criar máscara Bayer
    bayer_mask = np.zeros((height, width, 3), dtype=bool)
    
    for i in range(height):
        for j in range(width):
            pattern_idx = bayer_pattern[i % 2, j % 2]
            bayer_mask[i, j, pattern_idx] = True
    
    # Aplicar ruído diferente em cada posição do padrão
    noisy = image.copy()
    
    for c in range(3):
        channel_mask = bayer_mask[:, :, c]
        
        # Ruído específico para este canal no padrão Bayer
        noise = np.random.randn(height, width) * noise_levels[c]
        
        # Suavizar ligeiramente o ruído
        noise_smooth = ndimage.gaussian_filter(noise, sigma=0.5)
        
        # Aplicar apenas onde este canal está presente no padrão Bayer
        noisy_channel = noisy[:, :, c]
        noisy_channel[channel_mask] += noise_smooth[channel_mask]
        noisy[:, :, c] = np.clip(noisy_channel, 0, 1)
    
    # Simular imperfeições de demosaicking
    # Criar artefatos de zipper (bordas coloridas)
    edges = cv2.Canny((noisy.mean(axis=2) * 255).astype(np.uint8), 50, 150)
    edge_positions = np.where(edges > 0)
    
    for i, j in zip(edge_positions[0], edge_positions[1]):
        if i < height-1 and j < width-1:
            # Adicionar pequenas imperfeições nas bordas
            noisy[i, j, np.random.randint(0, 3)] += np.random.uniform(-0.05, 0.05)
    
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

# ==============================
# 5. RUÍDO DEPENDENTE DA COR
# ==============================

def add_color_dependent_noise(image: np.ndarray,
                         blue_noise_factor: float = 2.0,
                         red_noise_factor: float = 0.7) -> np.ndarray:
    """
    Ruído que varia com a cor (canal azul tipicamente mais ruidoso)
    
    Args:
        image: Imagem RGB
        blue_noise_factor: Fator de ruído para canal azul
        red_noise_factor: Fator de ruído para canal vermelho
    
    Returns:
        Imagem com ruído dependente da cor
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    height, width = image.shape[:2]
    
    # Fatores por canal
    noise_factors = [red_noise_factor, 1.0, blue_noise_factor]  # R, G, B
    
    noisy = image.copy()
    
    for c in range(3):
        # Ruído base
        noise = np.random.randn(height, width) * 0.02 * noise_factors[c]
        
        # Mais ruído em áreas escuras (fótons de menor energia)
        dark_mask = image[:, :, c] < 0.3
        noise[dark_mask] *= 2.0
        
        # Mais ruído em áreas de cor pura
        color_purity = np.abs(image[:, :, c] - image.mean(axis=2))
        purity_mask = color_purity > 0.4
        noise[purity_mask] *= 1.5
        
        noisy[:, :, c] = np.clip(noisy[:, :, c] + noise, 0, 1)
    
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

# ==============================
# 6. RUÍDO DE BALANCEAMENTO DE BRANCO
# ==============================

def add_white_balance_noise(image: np.ndarray,
                       temperature_shift: float = 1000.0,
                       tint_shift: float = 0.1) -> np.ndarray:
    """
    Simula erros de balanceamento de branco que amplificam ruído
    
    Args:
        image: Imagem RGB
        temperature_shift: Mudança de temperatura de cor (Kelvin)
        tint_shift: Mudança de matiz
    
    Returns:
        Imagem com ruído de balanceamento de branco
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # Primeiro, adicionar ruído básico
    noisy = add_color_dependent_noise((image * 255).astype(np.uint8), 1.5, 0.8) #AQUI
    noisy = noisy.astype(np.float32) / 255.0
    
    # Simular ganhos de balanceamento de branco
    # Luz quente (incandescente) → mais azul necessário
    if temperature_shift > 0:
        # Aumentar ganho do azul
        gain_b = 1.0 + temperature_shift / 4000.0
        gain_r = 1.0 / (1.0 + temperature_shift / 8000.0)
    else:
        # Luz fria → mais vermelho necessário
        gain_r = 1.0 + abs(temperature_shift) / 4000.0
        gain_b = 1.0 / (1.0 + abs(temperature_shift) / 8000.0)
    
    # Aplicar ganhos
    noisy[:, :, 0] *= gain_r  # Vermelho
    noisy[:, :, 2] *= gain_b  # Azul
    
    # Adicionar shift de matiz
    noisy[:, :, 0] += tint_shift * 0.1  # Vermelho
    noisy[:, :, 1] -= tint_shift * 0.05  # Verde
    noisy[:, :, 2] -= tint_shift * 0.05  # Azul
    
    # O balanceamento de branco amplifica o ruído existente
    # Adicionar ruído extra proporcional ao ganho
    extra_noise = np.random.randn(*noisy.shape) * 0.01 * np.array([gain_r, 1.0, gain_b])[np.newaxis, np.newaxis, :]
    noisy += extra_noise
    
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

# ==============================
# 7. RUÍDO DE CROSS-TALK ENTRE CANAIS
# ==============================

def add_cross_talk_noise(image: np.ndarray,
                    crosstalk_matrix: Optional[np.ndarray] = None,
                    edge_bleeding: bool = True) -> np.ndarray:
    """
    Simula vazamento entre canais de cor adjacentes
    
    Args:
        image: Imagem RGB
        crosstalk_matrix: Matriz 3x3 de cross-talk
        edge_bleeding: Se True, adiciona sangramento em bordas
    
    Returns:
        Imagem com cross-talk
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # Matriz de cross-talk padrão (cada linha soma ~1)
    if crosstalk_matrix is None:
        crosstalk_matrix = np.array([
            [0.95, 0.03, 0.02],  # Vermelho
            [0.04, 0.92, 0.04],  # Verde
            [0.02, 0.03, 0.95]   # Azul
        ])
    
    height, width = image.shape[:2]
    
    # Aplicar cross-talk linear
    noisy_flat = image.reshape(-1, 3) @ crosstalk_matrix.T
    noisy = noisy_flat.reshape(height, width, 3)
    
    # Adicionar sangramento em bordas (edge bleeding)
    if edge_bleeding:
        edges = cv2.Canny((image.mean(axis=2) * 255).astype(np.uint8), 50, 150)
        
        for c in range(3):
            channel = noisy[:, :, c].copy()
            
            # Dilatar bordas para simular sangramento
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Em bordas, misturar canais
            edge_mask = (edges_dilated > 0).astype(np.float32)
            
            # Para cada canal, misturar com canais vizinhos nas bordas
            other_channels = [noisy[:, :, (c+1)%3], noisy[:, :, (c+2)%3]]
            bleeding = sum(other_channels) / 2
            
            # Aplicar sangramento
            channel = channel * (1 - edge_mask*0.3) + bleeding * edge_mask*0.3
            noisy[:, :, c] = channel
    
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

# ==============================
# 8. RUÍDO DE COMPRESSÃO COLORIDA (JPEG)
# ==============================

def add_jpeg_color_artifacts(image: np.ndarray,
                        quality: int = 30,
                        subsampling: str = '4:2:0') -> np.ndarray:
    """
    Simula artefatos de compressão JPEG em imagens coloridas
    
    Args:
        image: Imagem RGB
        quality: Qualidade JPEG (1-100)
        subsampling: Subamostragem cromática
    
    Returns:
        Imagem com artefatos JPEG
    """
    # Usar OpenCV para compressão/descompressão JPEG real
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Converter para BGR para OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Comprimir e descomprimir
    result, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
    if not result:
        return image
    
    decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    # Converter de volta para RGB
    decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    
    # Adicionar artefatos específicos de cor
    height, width = decoded_rgb.shape[:2]
    
    # 1. Blocagem cromática (chroma blocking) - simplificado
    block_size = 8
    for i in range(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            if np.random.random() < 0.1:  # 10% dos blocos afetados
                channel_idx = np.random.randint(0, 3)
                block = decoded_rgb[i:i+block_size, j:j+block_size, channel_idx]
                
                # Adicionar viés de cor uniforme no bloco
                bias = np.random.uniform(-15, 15)
                block_float = block.astype(np.float32) + bias
                block_float = np.clip(block_float, 0, 255)
                
                decoded_rgb[i:i+block_size, j:j+block_size, channel_idx] = block_float.astype(np.uint8)
    
    # 2. Bandas de cor em gradientes - versão simplificada e segura
    try:
        # Criar padrão de bandas vertical
        band_pattern = np.sin(np.linspace(0, 4*np.pi, height)) * 3
        band_pattern = band_pattern.reshape(-1, 1)  # Shape (height, 1)
        
        # Repetir para largura
        band_pattern_2d = np.repeat(band_pattern, width, axis=1)  # Shape (height, width)
        
        # Detectar áreas suaves
        gray = cv2.cvtColor(decoded_rgb, cv2.COLOR_RGB2GRAY)
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
        smooth_mask = gradient < 15
        
        # Aplicar bandas apenas em áreas suaves
        for c in range(3):
            channel = decoded_rgb[:, :, c].astype(np.float32)
            # Aplicar bandas com intensidade reduzida
            channel[smooth_mask] += band_pattern_2d[smooth_mask] * 0.3
            decoded_rgb[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
            
    except Exception as e:
        print(f"Aviso: Erro ao aplicar bandas JPEG: {e}")
        # Continuar sem as bandas
    
    return decoded_rgb

# ==============================
# 9. ABERRAÇÃO CROMÁTICA
# ==============================

def add_chromatic_aberration(image: np.ndarray,
                        aberration_strength: float = 2.0,
                        radial: bool = True) -> np.ndarray:
    """
    Adiciona aberração cromática (bordas coloridas)
    
    Args:
        image: Imagem RGB
        aberration_strength: Força da aberração
        radial: Se True, aberração radial; se False, lateral
    
    Returns:
        Imagem com aberração cromática
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    height, width = image.shape[:2]
    
    # Centro da imagem
    center_y, center_x = height / 2, width / 2
    
    # Deslocamentos diferentes para cada canal
    shifts = {
        'r': (-aberration_strength * 1.0, -aberration_strength * 0.5),  # Vermelho
        'g': (0, 0),  # Verde (referência)
        'b': (aberration_strength * 0.8, aberration_strength * 0.3)   # Azul
    }
    
    channels = cv2.split(image)
    aberrated_channels = []
    
    for idx, (channel, color) in enumerate(zip(channels, ['r', 'g', 'b'])):
        dx, dy = shifts[color]
        
        if radial:
            # Aberração radial (mais forte nas bordas)
            # Criar mapa de deslocamento radial
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # Distância normalizada do centro
            dist_y = (y_coords - center_y) / center_y
            dist_x = (x_coords - center_x) / center_x
            distance = np.sqrt(dist_x**2 + dist_y**2)
            
            # Deslocamento proporcional à distância do centro
            shift_map_y = dy * distance * dist_y
            shift_map_x = dx * distance * dist_x
            
            # Coordenadas deslocadas
            coords_y = y_coords + shift_map_y
            coords_x = x_coords + shift_map_x
            
            # Interpolar canal
            aberrated = map_coordinates(channel, [coords_y, coords_x], 
                                      order=1, mode='reflect')
        else:
            # Aberração lateral simples (shift uniforme)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aberrated = cv2.warpAffine(channel, M, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        
        aberrated_channels.append(aberrated)
    
    aberrated_image = cv2.merge(aberrated_channels)
    return np.clip(aberrated_image * 255, 0, 255).astype(np.uint8)

# ==============================
# 10. FLARE COLORIDO
# ==============================

def add_color_flare(image: np.ndarray,
               flare_intensity: float = 0.3,
               flare_color: Tuple[float, float, float] = (1.0, 0.8, 0.6)) -> np.ndarray:
    """
    Adiciona flare colorido (reflexões internas na lente)
    
    Args:
        image: Imagem RGB
        flare_intensity: Intensidade do flare
        flare_color: Cor do flare (RGB)
    
    Returns:
        Imagem com flare
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    height, width = image.shape[:2]
    
    # Criar padrões de flare
    flare_layer = np.zeros_like(image)
    
    # Posições aleatórias para flares
    num_flares = np.random.randint(1, 4)
    
    for _ in range(num_flares):
        # Posição aleatória
        center_x = np.random.randint(width//4, 3*width//4)
        center_y = np.random.randint(height//4, 3*height//4)
        
        # Tamanho aleatório
        size = np.random.randint(min(height, width)//10, min(height, width)//4)
        
        # Criar flare circular com gradiente
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Perfil de intensidade do flare
        flare_strength = np.exp(-distance**2 / (2*(size/3)**2))
        flare_strength = flare_strength * flare_intensity * np.random.uniform(0.5, 1.0)
        
        # Adicionar anéis concêntricos para flare mais realista
        rings = np.sin(distance * 0.05) * 0.3 + 1.0
        flare_strength = flare_strength * rings
        
        # Aplicar cor ao flare
        for c in range(3):
            flare_layer[:, :, c] += flare_strength * flare_color[c]
    
    # Combinar flare com imagem (adição)
    flared_image = image + flare_layer
    
    # Adicionar veiling flare (redução de contraste global)
    veiling = np.ones_like(image) * flare_intensity * 0.1
    flared_image = flared_image * (1 - flare_intensity*0.2) + veiling
    
    return np.clip(flared_image * 255, 0, 255).astype(np.uint8)

# ==============================
# 11. RUÍDO DE DEMOSAICKING
# ==============================

def add_demosaicing_artifacts(image: np.ndarray,
                        artifact_strength: float = 0.5) -> np.ndarray:
    """
    Simula artefatos de demosaicking - VERSÃO SIMPLIFICADA E ROBUSTA
    """
    try:
        # Converter para float32 [0, 1] para processamento
        if image.dtype != np.float32:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.copy()
            if img_float.max() > 1.0:
                img_float = img_float / 255.0
        
        height, width = img_float.shape[:2]
        
        # 1. Simular padrão de Bayer simples
        # Criar versão com subamostragem Bayer
        bayer_sampled = np.zeros_like(img_float)
        
        for i in range(height):
            for j in range(width):
                # Padrão RGGB
                if i % 2 == 0:  # Linha par
                    if j % 2 == 0:  # Coluna par - Red
                        bayer_sampled[i, j, 0] = img_float[i, j, 0]
                        # Interpolar verde e azul
                        bayer_sampled[i, j, 1] = 0.5
                        bayer_sampled[i, j, 2] = 0.5
                    else:  # Coluna ímpar - Green
                        bayer_sampled[i, j, 1] = img_float[i, j, 1]
                        bayer_sampled[i, j, 0] = 0.5
                        bayer_sampled[i, j, 2] = 0.5
                else:  # Linha ímpar
                    if j % 2 == 0:  # Coluna par - Green
                        bayer_sampled[i, j, 1] = img_float[i, j, 1]
                        bayer_sampled[i, j, 0] = 0.5
                        bayer_sampled[i, j, 2] = 0.5
                    else:  # Coluna ímpar - Blue
                        bayer_sampled[i, j, 2] = img_float[i, j, 2]
                        bayer_sampled[i, j, 0] = 0.5
                        bayer_sampled[i, j, 1] = 0.5
        
        # 2. Aplicar filtro de interpolação pobre (cria zipper effect)
        from scipy.ndimage import uniform_filter
        
        demosaicked = bayer_sampled.copy()
        
        for c in range(3):
            # Aplicar filtro uniforme (simula interpolação bilinear pobre)
            channel_filtered = uniform_filter(demosaicked[:, :, c], size=3)
            
            # Adicionar ruído de interpolação
            interp_noise = np.random.randn(height, width) * 0.02 * artifact_strength
            channel_filtered += interp_noise
            
            # Aplicar mais forte em bordas (zipper effect)
            # Detectar bordas no canal de luminância
            luminance = np.mean(demosaicked, axis=2)
            edges = cv2.Canny((luminance * 255).astype(np.uint8), 50, 150) / 255.0
            
            # Aumentar artefatos nas bordas
            channel_filtered += edges * 0.05 * artifact_strength * (1 if c == 0 else -0.5 if c == 2 else 0)
            
            demosaicked[:, :, c] = np.clip(channel_filtered, 0, 1)
        
        # 3. Adicionar falsas cores em regiões de alta frequência
        # Calcular gradiente
        gray = cv2.cvtColor((demosaicked * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Identificar regiões de alta frequência
        high_freq_threshold = np.percentile(gradient_mag, 80)
        high_freq_mask = gradient_mag > high_freq_threshold
        
        # Misturar cores nestas regiões
        for i in range(height):
            for j in range(width):
                if high_freq_mask[i, j]:
                    # Escolher aleatoriamente para misturar cores
                    if np.random.random() < 0.3:
                        src_channel = np.random.randint(0, 3)
                        dst_channel = (src_channel + 1) % 3
                        mix_factor = np.random.uniform(0.1, 0.3)
                        demosaicked[i, j, dst_channel] = (
                            demosaicked[i, j, dst_channel] * (1 - mix_factor) +
                            demosaicked[i, j, src_channel] * mix_factor
                        )
        
        # Clip final
        demosaicked = np.clip(demosaicked, 0, 1)
        
        return (demosaicked * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Erro em demosaicing (versão simplificada): {e}")
        import traceback
        traceback.print_exc()
        return image
    
#%% 

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data

# ==========================================
# CARREGAR IMAGEM DE BIBLIOTECA
# ==========================================

image = data.astronaut()  # RGB uint8

print("\nImagem de biblioteca carregada com sucesso.\n")

# ==========================================
# FUNÇÃO DE TESTE
# ==========================================

def test_noise(name, func, *args, **kwargs):
    try:
        result = func(image.copy(), *args, **kwargs)

        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

        print(f"{name}: OK")
        return result

    except Exception as e:
        print(f"{name}: ERRO -> {e}")
        return image.copy()

# ==========================================
# APLICAR RUÍDOS (~50% intensidade)
# ==========================================

results = {
    "Original": image,
    "Chroma Noise": test_noise("Chroma Noise",
                               add_chroma_noise,
                               noise_strength=0.1,
                               correlation=0.3),

    "Color Misalignment": test_noise("Color Misalignment",
                                     add_color_misalignment,
                                     shift_range=(1.0, 1.0),
                                     rotation_range=(-0.25, 0.25)),

    "Color Moiré": test_noise("Color Moiré",
                              add_color_moire,
                              frequency_range=(0.1, 0.15),
                              amplitude_range=(0.15, 0.15)),

    "Bayer Noise": test_noise("Bayer Pattern Noise",
                              simulate_bayer_noise,
                              noise_levels=(0.03, 0.015, 0.04)),

    "Color Dependent": test_noise("Color Dependent Noise",
                                  add_color_dependent_noise,
                                  blue_noise_factor=1.5,
                                  red_noise_factor=0.85),

    "White Balance": test_noise("White Balance Noise",
                                add_white_balance_noise,
                                temperature_shift=500.0,
                                tint_shift=0.05),

    "Cross Talk": test_noise("Cross-Talk Noise",
                             add_cross_talk_noise,
                             edge_bleeding=True),

    "JPEG Artifacts": test_noise("JPEG Color Artifacts",
                                 add_jpeg_color_artifacts,
                                 quality=50),

    "Chromatic Aberration": test_noise("Chromatic Aberration",
                                       add_chromatic_aberration,
                                       aberration_strength=1.0),

    "Color Flare": test_noise("Color Flare",
                              add_color_flare,
                              flare_intensity=0.15),

    "Demosaicing": test_noise("Demosaicing Artifacts",
                              add_demosaicing_artifacts,
                              artifact_strength=0.5),
}

print("\nTodos os testes executados.\n")

# ==========================================
# PLOT COMPARATIVO
# ==========================================

plt.figure(figsize=(16, 12))

for i, (name, img) in enumerate(results.items()):
    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(name, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()