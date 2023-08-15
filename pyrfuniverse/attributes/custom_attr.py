import pyrfuniverse.attributes as attr
from pyrfuniverse.side_channel.side_channel import (
    IncomingMessage,
    OutgoingMessage,
)

import pyrfuniverse.utils.rfuniverse_utility as utility

def GrabRelease(kwargs: dict) -> OutgoingMessage:
    compulsory_params = ['id', 'grab']
    optional_params = []
    utility.CheckKwargs(kwargs, compulsory_params)
    msg = OutgoingMessage()

    msg.write_int32(kwargs['id'])
    msg.write_string('GrabRelease')
    msg.write_bool(kwargs['grab'])
    return msg

def Reset(kwargs: dict) -> OutgoingMessage:
    compulsory_params = ['id', 'reset']
    optional_params = []
    utility.CheckKwargs(kwargs, compulsory_params)
    msg = OutgoingMessage()

    msg.write_int32(kwargs['id'])
    msg.write_string('Reset')
    msg.write_bool(kwargs['reset'])
    return msg


class CustomAttr(attr.BaseAttr):
    """
    This is an example of custom attribute class, without actual functions.
    """

    def parse_message(self, msg: IncomingMessage) -> dict:
        """
        Parse messages. This function is called by internal function.

        Returns:
            Dict: A dict containing useful information of this class.

            data['custom_message']: A custom message
        """
        # 1. First, complete the message parsing in parent class.
        super().parse_message(msg)
        
        # 2. Read the message from Unity in order.
        # Note that the reading order here should align with 
        # the writing order in CollectData() of CustomAttr.cs.
        self.data['custom_message'] = msg.read_string()
        # self.data['grabbed']  = msg.read_bool()
        return self.data
    
    def GrabRelease(self, grab: bool):
        msg = OutgoingMessage()
        msg.write_int32(self.id)
        msg.write_string('GrabRelease')
        msg.write_bool(grab)
        self.env.instance_channel.send_message(msg)

    def Reset(self, reset: bool):
        msg = OutgoingMessage()
        msg.write_int32(self.id)
        msg.write_string('GrabRelease')
        msg.write_bool(reset)
        self.env.instance_channel.send_message(msg)

    # An example of new API
    def CustomMessage(self, message: str):
        # 1. Define an out-going message
        msg = OutgoingMessage()

        # 2. The first data written must be the unique id.
        msg.write_int32(self.id)

        # 3. The second data written must be the function name.
        # The keyword `CustomMessage` here should also be added
        # as a new branch in AnalysisMsg() in CustomAttr.cs.
        msg.write_string('CustomMessage')

        # 4. Write in the data.
        msg.write_string(message)

        self.env.instance_channel.send_message(msg)
