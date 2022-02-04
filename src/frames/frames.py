import json
import numpy as np

from copy import deepcopy
from src.tools.tools import read_from_file, write_to_file
from src.widgets.widgets import QFrame, QVBoxLayout, QGridLayout, QHBoxLayout, Text, Button, Input, Colors
from src.widgets.widgets import TextStyle, QMessageBox
from src.algorythm.q_learning import Parameters, QLearning

parameters = Parameters()


class InfoFrame(QFrame):
    """
    In this class we show a frame which includes a welcome text and brief description of the app.
    Also, we ask user to input some data about the shape of model, penalties and goal position.
    """

    def __init__(self, handler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handler

        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)

        # ====================== Column 0 ======================
        self.rows_count_label = Text("Number of Rows", style=TextStyle.title_2)
        self.rows_count_input = Input(
            placeholder="Between 1 and 9")

        self.columns_count_label = Text(
            "Number of Columns", style=TextStyle.title_2)
        self.columns_count_input = Input(
            placeholder="Between 1 and 9")

        self.action_reward_label = Text(
            "Action Reward", style=TextStyle.title_2)
        self.action_reward_input = Input(
            validator="float",
            placeholder="To prevent agent do unnecessary movements.")

        self.epsilon_label = Text("Epsilon", style=TextStyle.title_2)
        self.epsilon_input = Input(
            placeholder="Between 0 and 1", validator="float")

        self.load_last_try_btn = Button(
            label="Load and Insert Stored Inputs", command=self.load_last_try)

        # ====================== Column 1 ======================
        self.wall_penalty_label = Text("Wall Penalty", style=TextStyle.title_2)
        self.wall_penalty_input = Input(
            validator="float",
            placeholder="To prevent hitting the wall.")

        self.row_goal_label = Text(
            "Goal Position (X)", style=TextStyle.title_2)
        self.row_goal_input = Input(
            placeholder="Between 1 and 9")

        self.column_goal_label = Text(
            "Goal Position (Y)", style=TextStyle.title_2)
        self.column_goal_input = Input(
            placeholder="Between 1 and 9")

        self.step_label = Text("Step size", style=TextStyle.title_2)
        self.step_input = Input(
            placeholder="Between 0 and 1", validator="float")

        self.discount_label = Text("Discount value", style=TextStyle.title_2)
        self.discount_input = Input(
            placeholder="Between 0 and 1", validator="float")

        self.episode_count_label = Text(
            "Episode Count", style=TextStyle.title_2)
        self.episode_count_input = Input(
            placeholder="Number of games to learning.", validator="unlimit int")

        self.submit_info_btn = Button(label="Next", command=self.get_set_infos)

        # Col 0
        self.main_layout.addWidget(self.rows_count_label, 0, 0)
        self.main_layout.addWidget(self.rows_count_input, 1, 0)
        self.main_layout.addWidget(self.columns_count_label, 2, 0)
        self.main_layout.addWidget(self.columns_count_input, 3, 0)
        self.main_layout.addWidget(self.action_reward_label, 4, 0)
        self.main_layout.addWidget(self.action_reward_input, 5, 0)
        self.main_layout.addWidget(self.epsilon_label, 6, 0)
        self.main_layout.addWidget(self.epsilon_input, 7, 0)
        self.main_layout.addWidget(self.discount_label, 8, 0)
        self.main_layout.addWidget(self.discount_input, 9, 0)
        self.main_layout.addWidget(self.load_last_try_btn, 10, 0)
        # Col 1
        self.main_layout.addWidget(self.wall_penalty_label, 0, 1)
        self.main_layout.addWidget(self.wall_penalty_input, 1, 1)
        self.main_layout.addWidget(self.row_goal_label, 2, 1)
        self.main_layout.addWidget(self.row_goal_input, 3, 1)
        self.main_layout.addWidget(self.column_goal_label, 4, 1)
        self.main_layout.addWidget(self.column_goal_input, 5, 1)
        self.main_layout.addWidget(self.step_label, 6, 1)
        self.main_layout.addWidget(self.step_input, 7, 1)
        self.main_layout.addWidget(self.episode_count_label, 8, 1)
        self.main_layout.addWidget(self.episode_count_input, 9, 1)
        self.main_layout.addWidget(self.submit_info_btn, 10, 1)

    def get_set_infos(self) -> None:
        """
        This method get all information which user added.
        """
        try:
            parameters.rows_count = int(self.rows_count_input.get_text())
            parameters.columns_count = int(self.columns_count_input.get_text())
            parameters.action_reward = float(
                self.action_reward_input.get_text())
            parameters.wall_penalty = float(self.wall_penalty_input.get_text())
            parameters.x_goal = int(self.row_goal_input.get_text())
            parameters.y_goal = int(self.column_goal_input.get_text())
            parameters.epsilon = parameters.init_epsilon = float(
                self.epsilon_input.get_text())
            parameters.alpha = float(self.step_input.get_text())
            parameters.gamma = float(self.discount_input.get_text())
            parameters.episode_count = int(self.episode_count_input.get_text())

            # A callback function which came from MainFrame we call it from here to destroy this
            # frame and call next frame.
            self.handler()

        except ValueError as e:
            msg = QMessageBox()
            msg.setText("Please fill all entries.")
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()

    def load_last_try(self) -> None:
        """
        Open data.json and gets its data then pass them to handler.
        """

        try:
            # get data from file
            data = read_from_file()
            # add them to Inputs
            self.set_loaded_data(data)

        except FileNotFoundError as e:
            msg = QMessageBox()
            msg.setText("There is no saved data!")
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()

        except json.JSONDecodeError as e:
            msg = QMessageBox()
            msg.setText(
                "Invalid JSON file.\nMaybe the stored data is removed. Please fill inputs manually.")
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()

    def set_loaded_data(self, data: dict) -> None:
        """
        After loading data from json file, they fill in related
        inputs.
        """
        self.rows_count_input.setText(str(data["rows_count"]))
        self.columns_count_input.setText(str(data["columns_count"]))
        self.action_reward_input.setText(str(data["action_reward"]))
        self.wall_penalty_input.setText(str(data["wall_penalty"]))
        self.epsilon_input.setText(str(data["epsilon"]))
        self.step_input.setText(str(data["alpha"]))
        self.discount_input.setText(str(data["gamma"]))
        self.row_goal_input.setText(str(data["x_goal"]))
        self.column_goal_input.setText(str(data["y_goal"]))
        self.episode_count_input.setText(str(data["episode_count"]))
        parameters.model = data["model"]

        # Disable these inputs help us to prevent user to manipulate current model shape
        self.rows_count_input.setDisabled(True)
        self.columns_count_input.setDisabled(True)
        self.row_goal_input.setDisabled(True)
        self.column_goal_input.setDisabled(True)


class ModelFrame(QFrame):
    """
    After user determine some initial data, we show him a model of
    environment based on data which user added. Here user should determine
    each states values.
    """

    def __init__(self, train_and_result: object, *args, **kwargs):
        """
        train_and_result is a callback function which came from MainFrame to
        start algorythm, get result, remove this frame and add ResultFrame.
        """
        super().__init__(*args, **kwargs)
        self.train_and_result = train_and_result

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.grid_layout = QGridLayout()

        self.description_label = Text(
            label="This is your model of environment. In each state you can add a number as a reward to help agent reach the goal." +
            "\nThe green node is your goal that you have to give it a large reward e.g. 100. You can leave other states, they will get 0.\n",
            style=TextStyle.paragraph_1)
        self.main_layout.addWidget(self.description_label)

        self.main_layout.addLayout(self.grid_layout)
        self.add_states(parameters.rows_count, parameters.columns_count)

        self.submit_model_button = Button(
            label="Train and Result", command=self.submit_model)
        self.main_layout.addWidget(self.submit_model_button)

    def add_states(self, rows_count, columns_count) -> None:
        """
        This method add Input widgets for each state to user determine the values
        of the states.
        rows_count and columns count are the values that user added in previous frame.
        """
        for i in range(rows_count):
            for j in range(columns_count):
                # check if any model exists, add the corresponding state value to widget
                if parameters.model != None:
                    setattr(self, f"state_{i}_{j}", Input(
                        validator="float"))
                    widget = getattr(self, f"state_{i}_{j}")
                    widget.set_text(parameters.model[i][j])
                # if any model does not exist, just add an blank Input widget
                else:
                    setattr(self, f"state_{i}_{j}", Input(validator="float"))

                    widget = getattr(self, f"state_{i}_{j}")
                self.grid_layout.addWidget(widget, i, j)

        # Get goal widget to assign specific style
        goal_widget = getattr(
            self, f"state_{parameters.x_goal-1}_{parameters.y_goal-1}")
        goal_widget.setStyleSheet(f"""
            color:{Colors.white};
            border:2px solid  {Colors.green};
            border-radius:2;
            font-size:14px;
            height:40px;
            margin-bottom:10px;
            padding:2px 3px;
        """)

    def submit_model(self) -> None:
        """
        This method collect all values of states and add them to the model.
        Then call train_and_result() method which came from MainFrame to start 
        the training.
        data shape should be something like this:
        """
        # first if any model doesn't exist, we make a zero model based on rows count and columns count
        parameters.make_model()
        for i in range(parameters.rows_count):
            r = np.zeros((parameters.columns_count))
            for j in range(parameters.columns_count):
                widget = getattr(self, f"state_{i}_{j}")
                try:
                    r[j] = float(widget.get_text())
                except Exception as e:
                    # if user leave any uf fields empty we add 0 for that state.
                    print("Exception", i, j)
                    widget.setText("0")
                    r[j] = 0
            parameters.model[i] = r
        parameters.model = np.array(parameters.model)
        # call callback function
        self.train_and_result()


class ResultFrame(QFrame):
    """
    After previous steps, now is time to show the result!
    In this class we make label widgets and show the result
    which derived from q_learning.
    """

    def __init__(self, result, restart: object, *args, **kwargs):
        """
        result:
            Contains the result of q_learning which MainFrame pass it to this class.
        restart:
            A callback function to destroy this class and add InfoFrame to start 
            from beginning
        """
        super().__init__(*args, **kwargs)

        self.result = result
        # Set Main Layout (GRID)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.result_label = Text(label="Result", style=TextStyle.title_1)
        self.main_layout.addWidget(self.result_label)

        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)

        self.show_result(policy=self.result["policy"])

        self.horizontal_layout = QHBoxLayout()
        self.main_layout.addLayout(self.horizontal_layout)

        self.save_button = Button(
            label="Save Inputs", command=self.save_inputs)
        self.horizontal_layout.addWidget(self.save_button)

        self.restart_button = Button(
            label="Restart", command=restart)
        self.horizontal_layout.addWidget(self.restart_button)

    def show_result(self, policy: list) -> None:
        """
        Loop through the rows and columns and make an Text widget to show
        in each state which decision is better.
        """
        # counter is for finding corresponding state in policy list.
        counter = 0
        for i in range(parameters.rows_count):
            for j in range(parameters.columns_count):
                # set attr Text widget to class
                setattr(self, f"state_{i}_{j}", Text(
                    label=f"{parameters.arrow_maker(policy[counter])}",
                    style=TextStyle.yellow_highlight)
                )
                # get added widget to add to layout
                widget = getattr(self, f"state_{i}_{j}")
                self.grid_layout.addWidget(widget, i, j)

                counter += 1
        # Get goal widget to assign specific style to it
        goal_widget = getattr(
            self, f"state_{parameters.x_goal-1}_{parameters.y_goal-1}")
        goal_widget.setStyleSheet(TextStyle.green_highlight)
        goal_widget.setText("Goal")

    def save_inputs(self) -> None:
        """
        This function call from Save Inputs button.
        Store all inputs from beginning and the results of trianing 
        which user inserted into json file for later use.
        """
        # use deepcopy to prevent manipulate Parameters  attributes
        data = deepcopy(parameters.__dict__)
        data.pop("actions")
        data["epsilon"] = parameters.init_epsilon
        data["x_goal"] = parameters.x_goal
        data["y_goal"] = parameters.y_goal
        data["rows_count"] = parameters.rows_count
        data["columns_count"] = parameters.columns_count
        data["alpha"] = parameters.alpha
        data["gamma"] = parameters.gamma
        data["model"] = [[j for j in i] for i in data["model"]]
        # write to json file
        write_to_file(data)
        msg = QMessageBox()
        msg.setText(
            "Data has been successfully saved..")
        msg.setWindowTitle("Saved")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()


class MainFrame(QFrame):
    """
    Core of this app is this class which in charge of manage and handle adding and removing 
    frames.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.result = None
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        welcome_text = Text(
            "Welcome to Reinforcement Learning Example App", style=TextStyle.title_1)
        self.main_layout.addWidget(welcome_text)
        developer = Text(
            "Design and Developed by Sina Karimi", style=TextStyle.paragraph_3)
        self.main_layout.addWidget(developer)

        description = Text(
            f"""In this app we want to interact with one of tabular Reinforcement Learning methods named Q-Learning.\nAs this is a tabular method, you just have only four actions, namely Left, Right, Up, Down, which we describe\nthem in this app as L, R, U, D respectively.\nFurthermore, you can determine your model and environment and consequently rewards, penalties and the goal.""",
            style=TextStyle.paragraph_2)

        self.info_frame = InfoFrame(handler=self.handler)

        self.main_layout.addWidget(description)
        self.main_layout.addWidget(self.info_frame)
        self.main_layout.addStretch()

    def handler(self):
        """
        This is a callback funciton which called from InfoFrame to
        destroy the InfoFrame and call initialize_model method
        to start making ModelFrame frame.
        """
        # remove info frame
        self.info_frame.deleteLater()
        delattr(self, "info_frame")

        # call this method to create next frame(Model Frame)
        self.initialize_model()

    def initialize_model(self) -> None:
        """
        This method in charge of making ModelFrame.
        """
        self.model_frame = ModelFrame(self.train_and_result)
        self.main_layout.addWidget(self.model_frame)
        self.main_layout.addStretch()

    def train_and_result(self):
        """
        After collecting all required data, its time to
        begin training. First we remove the ModelFrame.
        """
        # remove model frame
        self.model_frame.deleteLater()

        # Initialize QLearning which contains the RinforcementLearning process for training.
        self.q_learning = QLearning(parameters=parameters)
        self.q_learning.train()
        self.result = self.q_learning.get_results()

        # initialize result frame to show result in GUI
        self.result_frame = ResultFrame(
            result=self.result, restart=self.restart)
        self.main_layout.addWidget(self.result_frame)
        self.main_layout.addWidget(self.result_frame)
        self.main_layout.addStretch()

    def restart(self) -> None:
        """
        This is a callback function which call from ResultFrame
        to remove ResultFrame and add InfoFrame again to start
        at first.
        """
        self.result_frame.deleteLater()
        self.info_frame = InfoFrame(handler=self.handler)
        self.main_layout.addWidget(self.info_frame)
        self.main_layout.addStretch()
