
\#📄 `README.md`


\# 🧠 Deep Learning for Computer Vision (Docker Environment)


This repository provides a complete Docker-based environment for deep learning projects focused on computer vision using \*\*Python\*\*, \*\*Keras\*\*, and \*\*OpenCV\*\*.


It is designed to ensure package compatibility, avoid local dependency conflicts, and provide a consistent development environment across systems.

---

\## 📦 Pre-installed Packages

\- Python 3.11

\- TensorFlow 2.16.1

\- Keras

\- OpenCV (`opencv-contrib-python`)

\- scikit-learn

\- NumPy, pandas, matplotlib, seaborn

\- JupyterLab

---

\## 💻 Requirements

Before you begin, please ensure that the following are installed on your system:

\- \[Docker Desktop](https://www.docker.com/products/docker-desktop)

&nbsp; - For \*\*Windows 10/11 Home\*\*, Docker Desktop requires \*\*WSL 2\*\* (the installer guides you through the setup)

\- Terminal access (CMD, PowerShell, Git Bash, or Terminal on macOS/Linux)

---

\## ⚙️ Setup Instructions

\### 1. Clone or download the project

If you are using version control:

```bash

git clone https://github.com/Hic-Faketmez/opencv.git

cd opencv/6\_Deep\_Learning\_for\_CV

````

Or just ensure your `Dockerfile` and `deeplearning\_workspace/` are located in the same directory.

---

\### 2. Build the Docker Image


Use the provided `Dockerfile` to build the custom deep learning environment:


```bash

docker build -t deeplearning .

```

This command will create an image named `deeplearning`.

---

\### 3. Create and Run the Container

Use the following command to run the container and mount your project directory:

\#### On macOS/Linux or Git Bash:

```bash

docker run -it --name mydeeplearning -v ${PWD}/deeplearning\_workspace:/deeplearning deeplearning

```

\#### On Windows CMD:

```cmd

docker run -it --name mydeeplearning -v %cd%\\deeplearning\_workspace:/deeplearning deeplearning

```

This mounts the local folder `deeplearning\_workspace` into the container at `/deeplearning`.

---

\### 4. Work Inside the Container

Once the container starts, navigate to your working directory:

```bash

cd /deeplearning/src

python main.py

```

You can edit code from your host using VS Code, and run it from inside the container.

---

\## 📁 Recommended Directory Structure

```

6\_Deep\_Learning\_for\_CV/

├── Dockerfile

├── deeplearning\_workspace/

│   ├── src/

│   │   └── 6\_1\_Keras\_Basics.py

│   ├── data/

│   ├── notebooks/

│   └── models/

├── README.md

```

\* `src/` – Python scripts

\* `data/` – input datasets or raw files

\* `notebooks/` – Jupyter notebooks

\* `models/` – trained models

\* `README.md` – project-specific notes

---

\## 🧪 Using JupyterLab (Optional)

If you want to run JupyterLab inside the container:

1\. Add `-p 8888:8888` to the `docker run` command:

```bash

docker run -it -p 8888:8888 --name mydeeplearning \\

&nbsp; -v ${PWD}/deeplearning\_workspace:/deeplearning \\

&nbsp; deeplearning

```

2\. Inside the container:

```bash

cd /deeplearning

jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

```

3\. Open Jupyter from your browser using the token printed in the terminal:

```

http://localhost:8888/?token=YOUR\_TOKEN

```

---

\## 🔧 Useful Docker Commands

| Task              | Command                           |

| ----------------- | --------------------------------- |

| Stop container    | `docker stop mydeeplearning`      |

| Restart container | `docker start -ai mydeeplearning` |

| Remove container  | `docker rm mydeeplearning`        |

| Remove image      | `docker rmi deeplearning`         |

| List containers   | `docker ps -a`                    |

| List images       | `docker images`                   |

---

\## ⚠️ Known Limitations

\* Docker Desktop on \*\*Windows/macOS does not support direct access to camera or microphone\*\*.

\* If you need webcam or microphone access, consider running your scripts on your \*\*native Python environment\*\* instead of Docker.

\* You can still process image/audio/video files inside the container by mounting them into `deeplearning\_workspace`.

---

\## 📝 License \& Contributions

Feel free to fork this repository, improve it, or open pull requests.

If you have ideas or suggestions for improvement, contributions are welcome!

---

\## 🙋‍♂️ Author

Created by \*\*\\\[Xamax]\*\*

Contact: \[hicfarketmezfarkeder@gmail.com](mailto:hicfarketmezfarkeder@gmail.com)

GitHub: \[github.com/Hic-Faketmez](https://github.com/Hic-Faketmez)

```