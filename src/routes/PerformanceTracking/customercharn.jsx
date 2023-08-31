import { useState } from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import DataTable2 from "../../components/DataTable2";
import Sidepanel from "../../components/sidepanel";
import Modal from "../DeliveryRoutes/removePopup";
import bg from '../../images/mainbg1.jpg';





function CustomerCharn() {
  const [showPopup, setShowPopup] = useState(false);

  const handleRemoveButtonClick = () => {
    setShowPopup(true);
  };


  const data = [
    {
      id: 1,
      sendname: 'John Doe',
      senderaddress: 'test',
      province: "test",
      lastitem: "test",
      feedback: "test",
      charnstatus: "test",
      // other: "<a href='/UpdateOrder/1'><button class='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-2 rounded'>Update</button ></a>  <button class='bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-2 rounded' onclick=\"setShowPopup(true)\">Remove</button>  <a href='/MoreInfo/1'><button class='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-10 rounded'>More_Info</button></a>"
    },
    {
      id: 2,
      sendname: 'Jane Smith',
      senderaddress: 'test',
      province: "test2",
      lastitem: "test2",
      feedback: "test2",
      charnstatus: "test2",
    //   days: "test2",
      // other: "<a href='/UpdateOrder/2'><button class='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-2 rounded'>Update</button ></a>  <button class='bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-2 rounded' onclick=\"setShowPopup(true)\">Remove</button>  <a href='/MoreInfo/2'><button class='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-10 rounded'>More_Info</button></a>"
    },
    // Add more data as needed)
  ];

  return (
<div>


    <div className="flex h-screen w-screen main-body">
      <div className="flex flex-row w-auto h-screen main-body-container">
        <Sidepanel />

      </div>
      <div className="w-full">
        
        <div className="flex items-center justify-center pt-10 pb-5 text-xl uppercase">
          <h1 className="flex items-center justify-center pt-4 text-xl uppercase font-bold pb-4 mt-4 ml-64">
            Customer Churn
          </h1>

        </div>

        <div className="max-w-[90%] ml-[5%]">
          <DataTable2 data={data} className="w-[90%]"/> 
        {showPopup && <Modal />} 
        {/* Render the Modal component when showPopup state is true */}
        </div>
        

      </div>



    </div>
</div>

  );
}


export default CustomerCharn;