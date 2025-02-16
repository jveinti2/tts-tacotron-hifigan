# TTS Tacotron2 HiFi-GAN

Este repositorio contiene una aplicación local para generar texto a voz (Text-to-Speech, TTS) utilizando modelos preentrenados de Tacotron2 y HiFi-GAN.

---

## Requisitos

- GPU NVIDIA compatible con CUDA.

---

## Instalación y Ejecución

### 1. Instalación de dependencias

Ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### 2. Clonar repositorios

Clona los siguientes repositorios y crea un archivo `__init__.py` en la raíz de cada uno:

```bash
git clone --recursive https://github.com/rmcpantoja/tacotron2.git
git clone --recursive https://github.com/justinjohn0306/hifi-gan.git
```

### 3. Descargar modelos preentrenados

-- modelo entrenado en pasos anteriores
-- MLPTTS

### 4. Ejecutar la aplicación

Ejecuta el siguiente comando para iniciar el programa:

```bash
fastapi dev
```

---

## Referencias y enlaces de interés

- [TTS original en Colab](https://colab.research.google.com/drive/1lRGlbiK2wUCm07BKIhjV3dKej7jV0s1y?usp=sharing#scrollTo=dLqNA_gpvVEu)
- [Colab de entrenamiento](https://colab.research.google.com/drive/1y1f4UOAkdYzDttUWEG0ri0E-3hE4wq4Q?usp=sharing)
- [Colab de transcripción](https://colab.research.google.com/drive/179oopKDixj00hjkJmtQ-UWTYZ4xT9cZj?usp=sharing)

---

## Interes

En el proceso de vivir en una instancia Ec2 con amazon linux y GPU nvidia, se hizo necesario seguir [guia de instalacion CUDA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-driver-types)
